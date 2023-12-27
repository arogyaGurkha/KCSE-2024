from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import argparse

# load model and scheduler
DEVICE = "cuda"

def initialize_pipeline(model_id):
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to(DEVICE)
    return pipeline

def upscale(image, pipeline):
    low_res_img = image.resize((128, 128))

    prompt = "an ambulance"

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    return upscaled_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Image Upscaler')
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()
    
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = initialize_pipeline(model_id)

    image = Image.open(args.filename).convert("RGB")

    upscaled_image = upscale(image, pipeline)
    upscaled_image.save("upscaled.png")