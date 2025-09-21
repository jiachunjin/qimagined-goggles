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


@torch.no_grad()
def edit_with_thinking():
    pipeline = QwenImageEditPipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image-Edit")
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=None)

    while True:
        prompt = input("请输入prompt: ")
        img_path = input("img path:")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("加载图片出错，重新加载")
            continue

        # ----- preprocess the image -----
        calculated_height = 1088
        calculated_width = 960

        image = pipeline.image_processor.resize(image_raw, calculated_height, calculated_width)
        prompt_image = image
        image = pipeline.image_processor.preprocess(image, calculated_height, calculated_width)
        image = image.unsqueeze(2)
        # ----- generate prompt embedding for both the text and image -----
        template = pipeline.prompt_template_encode
        drop_idx = pipeline.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        print("my text:", txt)
        model_inputs = pipeline.processor(
            text=txt,
            images=prompt_image,
            padding=True,
            return_tensors="pt",
        ).to(device)

    # ----- continue the generation -----
    generation_output = pipeline.text_encoder.generate(
        **model_inputs,
        max_new_tokens=512,
        output_hidden_states=True,
        return_dict_in_generate=True,
        output_scores=True
    )
    generated_ids = generation_output.sequences
    all_hidden_states = generation_output.hidden_states
    last_layer_hidden_states = []
    for token_hidden_states in all_hidden_states:
        last_layer_hidden_states.append(token_hidden_states[-1])
    final_hidden_states = torch.cat(last_layer_hidden_states, dim=1)

    print(final_hidden_states.shape)

    # ----- print一下qwenvl生成了些什么东西 -----
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = pipeline.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("生成的文本:", output_text)

    # ----- 进一步处理hidden states得到续写的东西 -----
    split_hidden_states = [e[drop_idx:] for e in final_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    print(prompt_embeds.shape)
    print(encoder_attention_mask.shape)

    inputs = {
        "image": image_raw,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": encoder_attention_mask,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save(f"edit_{prompt}.png")

if __name__ == "__main__":
    edit_with_pipeline()