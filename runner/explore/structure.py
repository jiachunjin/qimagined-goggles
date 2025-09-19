import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def text_encode():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # ---------- load text encoder ----------
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    text_encoder = text_encoder.to(device, dtype).eval()

    # ---------- encode prompt ----------
    prompt = ["生成一张祝小明生日快乐的贺卡", " "]
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = [template.format(e) for e in prompt]
    print(formatted_prompt)
    drop_idx = 34

    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # print(txt_tokens.input_ids[0].shape, tokenizer.decode(txt_tokens.input_ids[0]))
    print(txt_tokens.attention_mask)

    encoder_hidden_states = text_encoder(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    print(hidden_states.shape)

    chosen_hidden_states = hidden_states[:, drop_idx:]
    print(chosen_hidden_states.shape)

    chosen_mask = txt_tokens.attention_mask[:, drop_idx:]
    print(chosen_mask)

    return chosen_hidden_states, chosen_mask

def complete_pipeline():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---------- load mmdit ----------
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(device)

    chosen_hidden_states, chosen_mask = text_encode()

    image = pipe(
        prompt_embeds               = chosen_hidden_states[0],
        prompt_embeds_mask          = chosen_mask[0],
        negative_prompt_embeds      = chosen_hidden_states[1],
        negative_prompt_embeds_mask = chosen_mask[1],
        true_cfg_scale              = 5.0,
        num_inference_steps         = 50,
        height                      = 1024,
        width                       = 1024,
    ).images[0]

    image.save("generation_structure.png")

if __name__ == "__main__":
    text_encode()
    