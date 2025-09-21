import torch
import os
from PIL import Image
from diffusers import QwenImageEditPipeline

@torch.no_grad()
def edit_with_pipeline():
    pipeline = QwenImageEditPipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image-Edit")
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=None)
    while True:
        # 等待用户输入
        prompt = input("请输入prompt: ")
        img_path = input("img path:")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("加载图片出错，重新加载")
            continue
        if prompt == "exit":
            break

        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
            output_image.save("output_image_edit.png")
            print("image saved at", os.path.abspath("output_image_edit.png"))

if __name__ == "__main__":
    edit_with_pipeline()