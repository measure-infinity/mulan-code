import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, CLIPVisionModelWithProjection, AutoTokenizer, CLIPTextModelWithProjection, CLIPModel

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import shutil

import os

def compute_similarity(img_files, prompt):
    print(prompt)
    
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    vision_process = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    text_token = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")


    cos_similarity = torch.nn.CosineSimilarity()

    token_inputs = text_token([prompt], padding=True, truncation=True, max_length=77, return_tensors="pt")

    text_outputs = text_model(**token_inputs)
    text_embeds = text_outputs.text_embeds
    # print(text_embeds.norm(p=2, dim=-1, keepdim=True))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    # print(text_embeds.shape)

    all_sim = []
    for j in range(len(img_files)):
        image = Image.open(img_files[j])
        inputs = vision_process(images=image, return_tensors="pt")
        outputs = vision_model(**inputs)
        image_embeds = outputs.image_embeds
        # print(image_embeds.norm(p=2, dim=-1, keepdim=True))

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        img_sim = cos_similarity(text_embeds, image_embeds)
        # print(f'img-text: {cos_similarity(text_embeds, image_embeds)}')

        model_path = "liuhaotian/llava-v1.5-13b"
        llava_prompt = "Write a description for the image within 30 words."
        image_file = img_files[j]

        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": llava_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": 0.0,
            "num_beams": 1,
            "max_new_tokens": 512,
        })()
        llava_output = eval_model(args)
        print(llava_output)
        
        token_inputs_ = text_token([llava_output], padding=True, truncation=True, max_length=77, return_tensors="pt")

        text_outputs_ = text_model(**token_inputs_)
        text_embeds_ = text_outputs_.text_embeds
        # print(text_embeds_.norm(p=2, dim=-1, keepdim=True))
        text_embeds_ = text_embeds_ / text_embeds_.norm(p=2, dim=-1, keepdim=True)
        txt_sim = cos_similarity(text_embeds, text_embeds_)
        # print(f'text-text: {cos_similarity(text_embeds, text_embeds_)}')

        avg_sim = (img_sim + txt_sim) / 2.

        all_sim.append(avg_sim.item())
    
    all_sim = np.array(all_sim)
    print(f'similarity: {all_sim}')


    return all_sim.argmax()








