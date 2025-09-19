import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def text_encode(text_encoder=None):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    # ---------- load text encoder ----------
    if text_encoder is None:
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

def encode(prompt, qwenvl):
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
    device = qwenvl.device
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = [template.format(e) for e in prompt]
    drop_idx = 34
    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    encoder_hidden_states = qwenvl(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    chosen_hidden_states = hidden_states[:, drop_idx:]
    chosen_mask = txt_tokens.attention_mask[:, drop_idx:]

    return chosen_hidden_states, chosen_mask

def complete_pipeline():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---------- load mmdit ----------
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(device)

    # chosen_hidden_states, chosen_mask = text_encode(pipe.text_encoder)

    prompt = ["生成一张祝小明生日快乐的贺卡"]
    prompt_neg = [" "]

    prompt_embeds, prompt_embeds_mask = encode(prompt, pipe.text_encoder)

    prompt_embeds_neg, prompt_embeds_mask_neg = pipe._get_qwen_prompt_embeds(
        prompt                = prompt_neg,
        device                = device,
    )

    # print(chosen_hidden_states.shape, prompt_embeds.shape)

    image = pipe(
        prompt_embeds               = prompt_embeds,
        prompt_embeds_mask          = prompt_embeds_mask,
        negative_prompt_embeds      = prompt_embeds_neg,
        negative_prompt_embeds_mask = prompt_embeds_mask_neg,
        true_cfg_scale              = 5.0,
        num_inference_steps         = 10,
        height                      = 512,
        width                       = 512,
    ).images[0]

    image.save("generation_structure.png")

if __name__ == "__main__":
    complete_pipeline()
    