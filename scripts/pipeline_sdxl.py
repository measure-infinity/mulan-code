import argparse, os, sys, glob
import cv2
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

sys.path.append('../')

from query import gpt
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


from clip_similarity import compute_similarity
import shutil


import sdxl
from diffusers import StableDiffusionXLPipeline

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor


# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def generate(opt, model, curr_prompt='a black headphone is on the left of a green phone', att_idx=1, curr_obj=None, prev_obj=None, first=False, g_weight=70., g_steps=5, overlapping=False, ratio=0., merge=False):

    seed_everything(opt.seed)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = curr_prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    
    # curr_obj = 'stove_order_test'

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    if (not first) and (not merge):
        test_img = Image.open(f'{opt.outdir}/samples/{prev_obj.lower()}.png').convert('RGB')
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                if (not first) and (not merge):
                    test_image = np.array(test_img).astype(np.float32) / 255.0
                    test_image = test_image[None].transpose(0, 3, 1, 2)
                    test_image = torch.from_numpy(test_image)
                    test_image = 2.*test_image - 1.
                    print(test_image.shape)
                    test_encoding = model.encode_first_stage(test_image.to(torch.float16).to(0))
                    # test_encoding = model.get_first_stage_encoding(test_encoding)
                    test_encoding = 0.18215 * test_encoding.mode()


                if not overlapping:
                    att_mask = Image.open(f'{opt.outdir}/masks/{curr_obj.lower()}.png').convert('L')
                    att_mask = torch.from_numpy(np.array(att_mask))
                    att_mask = (att_mask / 255.).unsqueeze(0).unsqueeze(0).to(0)
                    if (not first) and (not merge):
                        att_mask1 = Image.open(f'{opt.outdir}/masks/{curr_obj.lower()}_1.png').convert('L')
                        att_mask1 = torch.from_numpy(np.array(att_mask1))
                        att_mask1 = (att_mask1 / 255.).unsqueeze(0).unsqueeze(0).to(0)
                    else:
                        att_mask1 = None
                else:
                    att_mask = Image.open(f'{opt.outdir}/masks/{curr_obj.lower()}_{ratio}.png').convert('L')
                    att_mask = torch.from_numpy(np.array(att_mask))
                    att_mask = (att_mask / 255.).unsqueeze(0).unsqueeze(0).to(0)
                    if (not first) and (not merge):
                        att_mask1 = Image.open(f'{opt.outdir}/masks/{curr_obj.lower()}_{ratio}_1.png').convert('L')
                        att_mask1 = torch.from_numpy(np.array(att_mask1))
                        att_mask1 = (att_mask1 / 255.).unsqueeze(0).unsqueeze(0).to(0)
                    else:
                        att_mask1 = None


                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        # print(prompts)
                        # print(data)
                        # os._exit(0)
                        uc = None
                        uc = model.get_learned_conditioning(batch_size * [""])
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        prompts = [curr_prompt] # ['a sink and a stove']
                        print(curr_prompt)
                        c = model.get_learned_conditioning(prompts)
                        # os._exit(0)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sampler.make_schedule(50, verbose=False)
                        # samples_ddim, _, all_samples_, all_x0 = sampler.sample(S=10,
                        #                                  conditioning=c[0].unsqueeze(0),
                        #                                  batch_size=opt.n_samples,
                        #                                  shape=shape,
                        #                                  verbose=False,
                        #                                  unconditional_guidance_scale=opt.scale,
                        #                                  unconditional_conditioning=uc,
                        #                                  eta=opt.ddim_eta,
                        #                                  x_T=start_code)
                        
                        alphas_prod_next = torch.from_numpy(np.append((sampler.model.alphas_cumprod.cpu().numpy()[sampler.ddim_timesteps])[1:], 0.0))
                        print(alphas_prod_next.shape)
                        # print(sampler.ddim_timesteps)
                        # os._exit(0)
                        # indices = list(range())
                        time_eps = {}
                        time_eps1 = None
                        if (not first) and (not merge):
                            img = test_encoding.detach().clone()
                            # img1 = test_encoding1.detach().clone()
                            time_eps[1] = img
                            # time_eps1[1] = img1
                            for i in tqdm(range(len(sampler.ddim_timesteps))):
                                t = torch.tensor([sampler.ddim_timesteps[i]] * test_encoding.shape[0], device=0).long()
                                pred_eps = sampler.model.apply_model(img.float(), t, None, None, [], [], [], uc)
                                ### pred_eps1 = sampler.model.apply_model(img1.float(), t, None, None, [], [], [], uc) # Refine
                                # time_eps[sampler.ddim_timesteps[i]] = pred_eps
                                # pred_eps = pipeline.unet(img.float(), t).sample
                                pred_xstart = (img.float() - pred_eps * (1 - sampler.model.alphas_cumprod[t].reshape(-1, 1, 1, 1)).sqrt().to(0)) / sampler.model.alphas_cumprod[t].sqrt().reshape(-1, 1, 1, 1).to(0)
                                ### pred_xstart1 = (img1.float() - pred_eps1 * (1 - sampler.model.alphas_cumprod[t].reshape(-1, 1, 1, 1)).sqrt().to(0)) / sampler.model.alphas_cumprod[t].sqrt().reshape(-1, 1, 1, 1).to(0) # Refine
                                # pred_xstart = pred_xstart.clamp(-1, 1)
                                img = alphas_prod_next[i].sqrt().reshape(-1, 1, 1, 1).to(0) * pred_xstart + (1 - alphas_prod_next[i].reshape(-1, 1, 1, 1)).sqrt().to(0) * pred_eps
                                ### img1 = alphas_prod_next[i].sqrt().reshape(-1, 1, 1, 1).to(0) * pred_xstart1 + (1 - alphas_prod_next[i].reshape(-1, 1, 1, 1)).sqrt().to(0) * pred_eps1 # Refine
                                
                                if i != 49:
                                    time_eps[sampler.ddim_timesteps[i+1]] = img
                                    # time_eps1[sampler.ddim_timesteps[i+1]] = img1 # Refine

                        samples_ddim, _, all_samples_, all_x0, all_attns = sampler.sample(S=50,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code, time_eps=time_eps, time_eps1=time_eps1, att_mask=att_mask, att_mask1=att_mask1, att_idx=att_idx, g_weight=g_weight, g_steps=g_steps)
                        
                        # print(len(all_attns))
                        # print(len(all_attns[0]))
                        # os._exit(0)
                        for i in range(16):
                            block_out = 0.
                            for j in range(len(all_attns)):
                                block_out = block_out + (all_attns[j])[i]
                            block_out = block_out / len(all_attns)
                            print(f'{block_out.max()}, {block_out.min()}, {block_out.shape}')
                            img_shape = int((block_out.shape[0]) ** 0.5)
                            os.makedirs(f'../save_attentions/block_{i}', exist_ok=True)
                            for k in range(15):
                                saved_img = block_out[:, k]
                                saved_img = saved_img.reshape(img_shape, img_shape).cpu().numpy()
                                saved_img = Image.fromarray((saved_img * 255).round().astype("uint8"))
                                saved_img = saved_img.resize((512, 512))
                                saved_img.save(f'../save_attentions/block_{i}/attention_{k}.png')
                        
                        # Generate true masks
                        true_mask = []
                        for block_num in [5, 6, 7, 8]:
                            t_mask = Image.open(f'../save_attentions/block_{block_num}/attention_{att_idx}.png')
                            t_mask = np.array(t_mask)
                            t_mask = t_mask / 255.
                            t_mask = t_mask > 0.05
                            true_mask.append(t_mask)
                        saved_true_mask = np.logical_or(true_mask[0], true_mask[1])
                        saved_true_mask = np.logical_or(saved_true_mask, true_mask[2])
                        saved_true_mask = np.logical_or(saved_true_mask, true_mask[3])
                        saved_true_mask = Image.fromarray((saved_true_mask.astype('int') * 255).astype('uint8'))
                        os.makedirs(f'{opt.outdir}/true_masks', exist_ok=True)
                        if not overlapping:
                            saved_true_mask.save(f'{opt.outdir}/true_masks/{curr_obj.lower()}.png')
                        else:
                            saved_true_mask.save(f'{opt.outdir}/true_masks/{curr_obj.lower()}_{ratio}.png')

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # img = put_watermark(img, wm_encoder)
                                if not overlapping:
                                    img.save(os.path.join(sample_path, f"{curr_obj.lower()}.png"))
                                else:
                                    img.save(os.path.join(sample_path, f"{curr_obj.lower()}_{ratio}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--prompt",
#         type=str,
#         nargs="?",
#         default="a refrigerator on the left, a stove on the right, and a sink in the middle",
#         help="the prompt to render"
#     )
#     parser.add_argument(
#         "--scene",
#         type=str,
#         default="",
#         help="the scene to render"
#     )
#     parser.add_argument(
#         "--outdir",
#         type=str,
#         nargs="?",
#         help="dir to write results to",
#         default="../evaluation_sdxl/200_test"
#     )
#     parser.add_argument(
#         "--skip_grid",
#         action='store_true',
#         help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
#     )
#     parser.add_argument(
#         "--skip_save",
#         action='store_true',
#         help="do not save individual samples. For speed measurements.",
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         type=int,
#         default=50,
#         help="number of ddim sampling steps",
#     )
#     parser.add_argument(
#         "--plms",
#         action='store_true',
#         help="use plms sampling",
#     )
#     parser.add_argument(
#         "--dpm_solver",
#         action='store_true',
#         help="use dpm_solver sampling",
#     )
#     parser.add_argument(
#         "--laion400m",
#         action='store_true',
#         help="uses the LAION400M model",
#     )
#     parser.add_argument(
#         "--fixed_code",
#         action='store_true',
#         help="if enabled, uses the same starting code across samples ",
#     )
#     parser.add_argument(
#         "--ddim_eta",
#         type=float,
#         default=0.0,
#         help="ddim eta (eta=0.0 corresponds to deterministic sampling",
#     )
#     parser.add_argument(
#         "--n_iter",
#         type=int,
#         default=1,
#         help="sample this often",
#     )
#     parser.add_argument(
#         "--H",
#         type=int,
#         default=1024,
#         help="image height, in pixel space",
#     )
#     parser.add_argument(
#         "--W",
#         type=int,
#         default=1024,
#         help="image width, in pixel space",
#     )
#     parser.add_argument(
#         "--C",
#         type=int,
#         default=4,
#         help="latent channels",
#     )
#     parser.add_argument(
#         "--f",
#         type=int,
#         default=8,
#         help="downsampling factor",
#     )
#     parser.add_argument(
#         "--n_samples",
#         type=int,
#         default=1,
#         help="how many samples to produce for each given prompt. A.k.a. batch size",
#     )
#     parser.add_argument(
#         "--n_rows",
#         type=int,
#         default=0,
#         help="rows in the grid (default: n_samples)",
#     )
#     parser.add_argument(
#         "--scale",
#         type=float,
#         default=7.5,
#         help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
#     )
#     parser.add_argument(
#         "--from-file",
#         type=str,
#         help="if specified, load prompts from this file",
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="../../stableDiffusion/v1-inference.yaml",
#         help="path to config which constructs model",
#     )
#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         default="../../stableDiffusion/sd-v1-4-full-ema.ckpt",
#         help="path to checkpoint of model",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="the seed (for reproducible sampling)",
#     )
#     parser.add_argument(
#         "--precision",
#         type=str,
#         help="evaluate at this precision",
#         choices=["full", "autocast"],
#         default="autocast"
#     )
#     args = parser.parse_args()
#     print(args.prompt)
#     # args.outdir = f"../exps/{(args.prompt[:-1]).replace(',', '').replace(' ', '_') if args.prompt[-1] == '.' else (args.prompt).replace(',', '').replace(' ', '_')}"
#     # os.makedirs(args.outdir, exist_ok=True)

#     if args.laion400m:
#         print("Falling back to LAION 400M model...")
#         args.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
#         args.ckpt = "models/ldm/text2img-large/model.ckpt"
#         args.outdir = "outputs/txt2img-samples-laion400m"


def mulan_sdxl(prompt="a black headphone is on the left of a green phone", seed=42, sd_model="stabilityai/stable-diffusion-xl-base-1.0", weight=120.):

    pipe = StableDiffusionXLPipeline.from_pretrained(
    sd_model, torch_dtype=torch.float16, cache_dir='../sd_models',
    )
    
    pipe = pipe.to("cuda")

    args = OmegaConf.load("../configs/sdxl_params.yaml")
    args.seed = seed
    args.weight = weight
    
    counter = 0
    if isinstance(prompt, str):
        test_prompts = [prompt]
    else:
        test_prompts = prompt
    for test_prompt in test_prompts:
        args.prompt = test_prompt
        args.outdir = f"../sdxl_outputs"
        counter += 1
        os.makedirs(args.outdir, exist_ok=True)
        
        # GPT Decomposition
        decompose_prompt = f'You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. You only need to list the objects in the description by painting order, from left to right. Do not list additional information other than the objects mentioned in the description. Description: {args.prompt}.'
        decompose_output = gpt(p_template=decompose_prompt)
        decompose_output = decompose_output.split('\n')
        print(decompose_output)
        if len(decompose_output) == 1:
            decompose_output = decompose_output[0].split(',')
            decompose_output = [i.strip().replace('.', '') for i in decompose_output]
        else:
            decompose_output = [i.split('.')[1].strip() for i in decompose_output]

        i = 0
        obj_1 = decompose_output[i]
        if not obj_1.startswith('a '):
            obj_1 = 'a ' + obj_1

        if len(args.scene) >= 1:
            first_prompt = args.scene + f" {obj_1}"
        else:
            first_prompt = obj_1
        # os._exit(0)
        if (first_prompt.startswith("'") and first_prompt.endswith("'")) or (first_prompt.startswith('"') and first_prompt.endswith('"')):
            first_prompt = first_prompt[1:-1]

        obj_name = gpt(p_template=f"Given the description '{first_prompt}', what is the object in the description? Just give the object name.")
        if len(obj_name.split(' ')) > 1:
            obj_name = (obj_name.split(' '))[-1]
        
        print([j.lower() for j in first_prompt.split(' ')])
        att_idx = [j.lower() for j in first_prompt.split(' ')].index(obj_name.lower())

        att_idx += 1

        obj_pos = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {args.prompt}. If I want to paint the {obj_name.lower()} in the painting firstly, where to put the {obj_name.lower()}? Choose from left, right, top, and bottom. You can make reasonable guesses. Give one answer.")
        print(obj_pos)
        if 'left' in obj_pos.lower() or 'right' in obj_pos.lower():
            obj_num = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {args.prompt}. How many non-overlapping objects are there in the horizontal direction? ONLY give the final number.")
        else:
            obj_num = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {args.prompt}. How many non-overlapping objects are there in the vertical direction? ONLY give the final number.")
        
        if len(obj_num) > 1:
            obj_num = 1
        elif int(obj_num) == 0:
            obj_num = 1
        
        os.makedirs(f'{args.outdir}/masks', exist_ok=True)
        mask = np.zeros((1024, 1024))
        
        if 'left' in obj_pos.lower():
            obj_pos = 'left'
            mask[:, :(1024//int(obj_num))] = 1.
        elif 'right' in obj_pos.lower():
            obj_pos = 'right'
            mask[:, (1024//int(obj_num)*(int(obj_num)-1)):] = 1.
        elif 'top' in obj_pos.lower():
            obj_pos = 'top'
            mask[:(1024//int(obj_num)), :] = 1.
        else:
            obj_pos = 'bottom'
            mask[(1024//int(obj_num)*(int(obj_num)-1)):, :] = 1.
        mask = Image.fromarray((mask * 255).astype('uint8'))
        mask.save(f'{args.outdir}/masks/{obj_name.lower()}.png')

        repeat_gen = 0
        while repeat_gen < 3:
            if repeat_gen == 1:
                g_weight = weight - 20.
                g_steps = 10
            elif repeat_gen == 2:
                g_weight = weight + 30.
                g_steps = 15
            else:
                g_weight = weight
                g_steps = 10


            prev_latents = None
            curr_latents = sdxl.generate(args, pipe, curr_prompt=first_prompt, att_idx=att_idx, curr_obj=obj_name, first=True, g_weight=g_weight, g_steps=g_steps, prev_latents=prev_latents)

            model_path = "liuhaotian/llava-v1.5-13b"
            prompt = f"Does the image contain {obj_name.lower()}? Only answer yes or no."
            image_file = f"{args.outdir}/samples/{obj_name.lower()}.png"

            args_llava = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": prompt,
                "conv_mode": None,
                "image_file": image_file,
                "sep": ",",
                "temperature": 0,
                "top_p": 0.0,
                "num_beams": 1,
                "max_new_tokens": 512,
                "device": 0,
            })()

            
            test = eval_model(args_llava)
            if 'yes' in test.lower():
                print(repeat_gen)
                break
            repeat_gen += 1
        # os._exit(0)
        
        # Left objects
        already_have = ""
        already_have = already_have + f"{obj_name.lower()} positioned in/on the {obj_pos.lower()} of the painting"
        prev_pos = obj_pos.lower()
        for j in range(i+1, len(decompose_output)):
            prev_obj_name = obj_name

            obj_name = gpt(p_template=f"Given the description 'a {decompose_output[j]}', what is the object in the description? Just give the object name.")
            if len(obj_name.split(' ')) > 1:
                obj_name = (obj_name.split(' '))[-1]
            print(obj_name)
            obj_pos = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {args.prompt}. If I already have a painting that contains {already_have}, what is the position of the {obj_name.lower()} relative to the {prev_obj_name.lower()}? Choose from left, right, above, bottom, and none of above. You can make reasonable guesses. Give one answer.")
            print(obj_pos)
            # os._exit(0)
            if 'none' in obj_pos.lower():
                obj_pos = 'none'
            elif 'left' in obj_pos.lower():
                obj_pos = 'left'
            elif 'right' in obj_pos.lower():
                obj_pos = 'right'
            elif 'above' in obj_pos.lower():
                obj_pos = 'above'
            else:
                obj_pos = 'bottom'
            
            if obj_pos == 'none':
                input_prompt = gpt(p_template=f"Given the description {args.prompt}, extract the substring that contains {prev_obj_name} and {obj_name}. Do not use punctuations.")
                already_have = already_have + f" and {input_prompt.lower()} positioned in/on the {prev_pos}"
            
            
            if obj_pos != 'none':
                obj_num = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {args.prompt}. If I already have a painting that contains {already_have}, how many objects are there in/on the {obj_pos.lower()} of {prev_obj_name.lower()}? Only give the final number.")
                if len(obj_num) > 1:
                    obj_num = 1
                elif int(obj_num) == 0:
                    obj_num = 1
                

                curr_substr = gpt(p_template=f"Given the description {args.prompt}, extract the substring that contains {prev_obj_name.lower()} and the {obj_name.lower()}. Do not use punctuations.")
                overlapping = gpt(p_template=f"You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {curr_substr}. Should be there any overlap between the {prev_obj_name.lower()} and the {obj_name.lower()}? Yes or no.")
                prev_true_mask = Image.open(f"{args.outdir}/true_masks/{prev_obj_name.lower()}.png")
                prev_true_mask = np.array(prev_true_mask) / 255.
                y_m, x_m = np.where(prev_true_mask == 1.)
                mask = np.zeros((1024, 1024))
                mask1 = np.zeros((1024, 1024))
                if 'no' in overlapping.lower():
                    already_have = already_have + f" and {obj_name.lower()} positioned in/on the {obj_pos.lower()} of the {prev_obj_name.lower()}"
                    if obj_pos == 'left':
                        mask[:, (x_m.min()//int(obj_num)*(int(obj_num)-1)):x_m.min()] = 1.
                        mask1[:, :x_m.min()] = 1.
                    elif obj_pos == 'right':
                        mask[:, x_m.max():(x_m.max()+(1024-x_m.max())//int(obj_num))] = 1.
                        mask1[:, x_m.max():] = 1.
                    elif obj_pos == 'above':
                        mask[(y_m.min()//int(obj_num)*(int(obj_num)-1)):y_m.min(), :] = 1.
                        mask1[:y_m.min(), :] = 1.
                    else:
                        mask[y_m.max():(y_m.max()+(1024-y_m.max())//int(obj_num)), :] = 1.
                        mask1[y_m.max():, :] = 1.
                    mask = Image.fromarray((mask * 255).astype('uint8'))
                    mask.save(f'{args.outdir}/masks/{obj_name.lower()}.png')
                    mask1 = Image.fromarray((mask1 * 255).astype('uint8'))
                    mask1.save(f'{args.outdir}/masks/{obj_name.lower()}_1.png')

                    input_prompt = f"a {decompose_output[j].lower()}"
                    if len(args.scene) >= 1:
                        input_prompt = "a realistic scene with" + f" {input_prompt}"
                    att_idx = [name.lower() for name in input_prompt.split(' ')].index(obj_name.lower())
                    att_idx += 1
                    repeat_gen = 0
                    prev_latents = curr_latents
                    while repeat_gen < 3:
                        if repeat_gen == 1:
                            g_weight = 80.
                            g_steps = 10
                        elif repeat_gen == 2:
                            g_weight = 100.
                            g_steps = 10
                        else:
                            g_weight = 80.
                            g_steps = 10


                        curr_latents = sdxl.generate(args, pipe, curr_prompt=input_prompt, att_idx=att_idx, curr_obj=obj_name, prev_obj=prev_obj_name, first=False, g_weight=g_weight, g_steps=g_steps, prev_latents=prev_latents)

                        model_path = "liuhaotian/llava-v1.5-13b"
                        prompt = f"Does the image contain {obj_name.lower()}? Only answer yes or no."
                        image_file = f"{args.outdir}/samples/{obj_name.lower()}.png"

                        args_llava = type('Args', (), {
                            "model_path": model_path,
                            "model_base": None,
                            "model_name": get_model_name_from_path(model_path),
                            "query": prompt,
                            "conv_mode": None,
                            "image_file": image_file,
                            "sep": ",",
                            "temperature": 0,
                            "top_p": 0.0,
                            "num_beams": 1,
                            "max_new_tokens": 512,
                            "device": 0,
                        })()

                        
                        test = eval_model(args_llava)
                        if 'yes' in test.lower():
                            print(repeat_gen)
                            break
                        repeat_gen += 1
                else:
                    print(f'overlapping between {prev_obj_name.lower()} and {obj_name.lower()}')
                    input_prompt = gpt(p_template=f"Given the description {args.prompt}, extract the substring that contains {prev_obj_name.lower()} and {obj_name.lower()}. Do not use punctuations.")
                    already_have = already_have + f" and {input_prompt.lower()}"
                    prev_latents = curr_latents
                    all_curr_latents = []
                    if len(args.scene) >= 1:
                        input_prompt = "a realistic scene with" + f" {input_prompt}"
                    for ratio in [0.1, 0.3, 0.5]:
                        mask = np.zeros((1024, 1024))
                        mask1 = np.zeros((1024, 1024))
                        mask_h = y_m.max() - y_m.min()
                        mask_w = x_m.max() - x_m.min()
                        if obj_pos == 'left':
                            mask[y_m.min():y_m.max(), (x_m.min()//int(obj_num)*(int(obj_num)-1)):(x_m.min()+round(mask_w*ratio))] = 1.
                            mask1[y_m.min():y_m.max(), :(x_m.min()+round(mask_w*ratio))] = 1.
                        elif obj_pos == 'right':
                            mask[y_m.min():y_m.max(), (x_m.max()-round(mask_w*ratio)):(x_m.max()+(1024-x_m.max())//int(obj_num))] = 1.
                            mask1[y_m.min():y_m.max(), (x_m.max()-round(mask_w*ratio)):] = 1.
                        elif obj_pos == 'above':
                            mask[(y_m.min()//int(obj_num)*(int(obj_num)-1)):(y_m.min()+round(mask_h*ratio)), x_m.min():x_m.max()] = 1.
                            mask1[:(y_m.min()+round(mask_h*ratio)), x_m.min():x_m.max()] = 1.
                        else:
                            mask[(y_m.max()-round(mask_h*ratio)):(y_m.max()+(1024-y_m.max())//int(obj_num)), x_m.min():x_m.max()] = 1.
                            mask1[(y_m.max()-round(mask_h*ratio)):, x_m.min():x_m.max()] = 1.
                        mask = Image.fromarray((mask * 255).astype('uint8'))
                        mask.save(f'{args.outdir}/masks/{obj_name.lower()}_{ratio}.png')
                        mask1 = Image.fromarray((mask1 * 255).astype('uint8'))
                        mask1.save(f'{args.outdir}/masks/{obj_name.lower()}_{ratio}_1.png')

                        att_idx = [name.lower() for name in input_prompt.split(' ')].index(obj_name.lower())
                        att_idx += 1
                        repeat_gen = 0
                        while repeat_gen < 3:
                            if repeat_gen == 1:
                                g_weight = 80.
                                g_steps = 10
                            elif repeat_gen == 2:
                                g_weight = 100.
                                g_steps = 10
                            else:
                                g_weight = 80.
                                g_steps = 10


                            curr_latents_ = sdxl.generate(args, pipe, curr_prompt=input_prompt, att_idx=att_idx, curr_obj=obj_name, prev_obj=prev_obj_name, first=False, g_weight=g_weight, g_steps=g_steps, overlapping=True, ratio=ratio, prev_latents=prev_latents)

                            model_path = "liuhaotian/llava-v1.5-13b"
                            prompt = f"Does the image contain {obj_name.lower()}? Only answer yes or no."
                            image_file = f"{args.outdir}/samples/{obj_name.lower()}_{ratio}.png"

                            args_llava = type('Args', (), {
                                "model_path": model_path,
                                "model_base": None,
                                "model_name": get_model_name_from_path(model_path),
                                "query": prompt,
                                "conv_mode": None,
                                "image_file": image_file,
                                "sep": ",",
                                "temperature": 0,
                                "top_p": 0.0,
                                "num_beams": 1,
                                "max_new_tokens": 512,
                                "device": 0,
                            })()

                            
                            test = eval_model(args_llava)
                            if 'yes' in test.lower():
                                print(repeat_gen)
                                break
                            repeat_gen += 1
                        
                        all_curr_latents.append(curr_latents_)
                    
                    # Select one
                    selected_num = compute_similarity([f"{args.outdir}/samples/{obj_name.lower()}_{ratio}.png" for ratio in [0.1, 0.3, 0.5]], already_have)
                    selected_ratio = ([0.1, 0.3, 0.5])[selected_num]
                    shutil.copyfile(f"{args.outdir}/samples/{obj_name.lower()}_{selected_ratio}.png", f"{args.outdir}/samples/{obj_name.lower()}.png")
                    shutil.copyfile(f"{args.outdir}/true_masks/{obj_name.lower()}_{selected_ratio}.png", f"{args.outdir}/true_masks/{obj_name.lower()}.png")
                    curr_latents = all_curr_latents[selected_num]
            else:
                att_idx = [j.lower() for j in input_prompt.split(' ')].index(obj_name.lower())
                att_idx += 1
                obj_name = input_prompt
                shutil.copyfile(f"{args.outdir}/masks/{prev_obj_name.lower()}.png", f"{args.outdir}/masks/{obj_name.lower()}.png")
                repeat_gen = 0
                prev_latents = curr_latents
                while repeat_gen < 3:
                    if repeat_gen == 1:
                        g_weight = 80.
                        g_steps = 10
                    elif repeat_gen == 2:
                        g_weight = 100.
                        g_steps = 10
                    else:
                        g_weight = 80.
                        g_steps = 10


                    curr_latents = sdxl.generate(args, pipe, curr_prompt=input_prompt, att_idx=att_idx, curr_obj=obj_name, first=False, g_weight=g_weight, g_steps=g_steps, merge=True, prev_latents=prev_latents)

                    model_path = "liuhaotian/llava-v1.5-13b"
                    prompt = f"Does the image contain {obj_name.lower()}? Only answer yes or no."
                    image_file = f"{args.outdir}/samples/{obj_name.lower()}.png"

                    args_llava = type('Args', (), {
                        "model_path": model_path,
                        "model_base": None,
                        "model_name": get_model_name_from_path(model_path),
                        "query": prompt,
                        "conv_mode": None,
                        "image_file": image_file,
                        "sep": ",",
                        "temperature": 0,
                        "top_p": 0.0,
                        "num_beams": 1,
                        "max_new_tokens": 512,
                        "device": 0,
                    })()

                    
                    test = eval_model(args_llava)
                    if 'yes' in test.lower():
                        print(repeat_gen)
                        break
                    repeat_gen += 1
            
            if j == (len(decompose_output) - 1):
                shutil.copyfile(f"{args.outdir}/samples/{obj_name.lower()}.png", f"{args.outdir}/samples/final_{obj_name.lower()}.png")


        

