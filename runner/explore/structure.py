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
    prompt = ["生成一张祝小明生日快乐的贺卡"]
    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    formatted_prompt = [template.format(e) for e in prompt]
    print(formatted_prompt)

    txt_tokens = tokenizer(
        formatted_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    print(txt_tokens.input_ids[0].shape, tokenizer.decode(txt_tokens.input_ids[0]))
    print(txt_tokens.attention_mask[0].shape)


    
if __name__ == "__main__":
    text_encode()
    