import sys
sys.path.append('../')

import torch
from diffusers import StableDiffusionXLPipeline
import os
from PIL import Image
import numpy as np
from pytorch_lightning import seed_everything
import random

def set_seed(seed = 0):
    np.random.seed(seed)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir='../sdxl_cache',
# )

# pipe = pipe.to("cuda")


def generate(opt, pipe, curr_prompt='a dog sitting on the ground and a cat sitting on a table', att_idx=1, curr_obj=None, prev_obj=None, first=False, g_weight=200., g_steps=10, overlapping=False, ratio=0., merge=False, prev_latents=None):

    set_seed(opt.seed)

    outpath = opt.outdir
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    latents = torch.randn(1, 4, 128, 128, dtype=torch.float16)

    device = 0

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

    prompt = curr_prompt

    all_attns = []


    output, curr_latents = pipe(prompt=prompt, latents=latents.detach().clone(), all_attns=all_attns, prev_latents=prev_latents, att_mask=att_mask, att_mask1=att_mask1, att_idx=att_idx, g_weight=g_weight, g_steps=g_steps)
    image = output.images[0]
    
    for i in range(40):
        block_out = 0.
        for j in range(len(all_attns)):
            block_out = block_out + (all_attns[j])[i]
        block_out = block_out / len(all_attns)
        print(f'{block_out.max()}, {block_out.min()}, {block_out.shape}')
        img_shape = int((block_out.shape[0]) ** 0.5)
        os.makedirs(f'../save_attentions_sdxl/block_{i}', exist_ok=True)
        for k in range(20):
            saved_img = block_out[:, k]
            saved_img = saved_img.reshape(img_shape, img_shape).cpu().numpy()
            saved_img = Image.fromarray((saved_img * 255).round().astype("uint8"))
            saved_img = saved_img.resize((1024, 1024))
            saved_img.save(f'../save_attentions_sdxl/block_{i}/attention_{k}.png')
    

    true_mask = []
    for block_num in [35, 36, 37]:
        t_mask = Image.open(f'../save_attentions_sdxl/block_{block_num}/attention_{att_idx}.png')
        t_mask = np.array(t_mask)
        t_mask = t_mask / 255.
        t_mask_ = t_mask.max() * (4 / 5)
        t_mask = t_mask > t_mask_
        true_mask.append(t_mask)
    saved_true_mask = np.logical_or(true_mask[0], true_mask[1])
    saved_true_mask = np.logical_or(saved_true_mask, true_mask[2])
    att_mask_ = att_mask.cpu().clone().squeeze().numpy()
    mask_y, mask_x = np.where(att_mask_==1.)
    saved_true_mask[:, :mask_x.min()] = 0.
    saved_true_mask = Image.fromarray((saved_true_mask.astype('int') * 255).astype('uint8'))
    os.makedirs(f'{opt.outdir}/true_masks', exist_ok=True)
    if not overlapping:
        saved_true_mask.save(f'{opt.outdir}/true_masks/{curr_obj.lower()}.png')
    else:
        saved_true_mask.save(f'{opt.outdir}/true_masks/{curr_obj.lower()}_{ratio}.png')

    print(type(image))
    if not overlapping:
        image.save(os.path.join(sample_path, f"{curr_obj.lower()}.png"))
    else:
        image.save(os.path.join(sample_path, f"{curr_obj.lower()}_{ratio}.png"))
    
    return curr_latents


