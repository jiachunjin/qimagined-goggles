from diffusers import QwenImageEditPipeline

@torch.no_grad()
def edit_with_pipeline():
    pipeline = QwenImageEditPipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image-Edit")
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda:1")
    pipeline.set_progress_bar_config(disable=None)
    image = Image.open("assets/letter1.webp").convert("RGB")
    prompt = '把信件上的中文去除掉，改写成: "Autumn Breeze, Letter Arrives"'
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