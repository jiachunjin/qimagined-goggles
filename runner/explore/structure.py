import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


def text_encode_bsz_1():
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

    # print(txt_tokens.input_ids[0].shape, tokenizer.decode(txt_tokens.input_ids[0]))
    # print(txt_tokens.attention_mask[0].shape)

    encoder_hidden_states = text_encoder(
        input_ids            = txt_tokens.input_ids,
        attention_mask       = txt_tokens.attention_mask,
        output_hidden_states = True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    print(hidden_states.shape)

    chosen_hidden_states = [e[drop_idx:] for e in hidden_states]
    print(chosen_hidden_states.shape)
    

    
if __name__ == "__main__":
    text_encode_bsz_1()
    