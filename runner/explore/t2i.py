"""
试一下qwen-image对system prompt的鲁棒性
"""
import torch
from diffusers import QwenImagePipeline
from transformers import AutoTokenizer, AutoProcessor

device = torch.device("cuda:6")
dtype = torch.bfloat16

# ---------- load models ----------
pipe = QwenImagePipeline.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/tokenizer")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
qwenvl = pipe.text_encoder

# ---------- load models ----------
prompt = ["生成一张祝小明生日快乐的贺卡"]
template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
txt = [template.format(e) for e in prompt]
print(txt)

prompt_tokens = tokenizer(
    txt, max_length=1024, padding=True, truncation=True, return_tensors="pt"
).to(device)
print(prompt_tokens.input_ids[0].shape, tokenizer.decode(prompt_tokens.input_ids[0]))

encoder_hidden_states = qwenvl(
    input_ids            = prompt_tokens.input_ids,
    attention_mask       = prompt_tokens.attention_mask,
    output_hidden_states = True,
)
hidden_states = encoder_hidden_states.hidden_states[-1]

prompt_embeds = hidden_states
prompt_embeds_mask = prompt_tokens.attention_mask

# ---------- call mmdit ----------
image = pipe(
    prompt_embeds       = prompt_embeds,
    prompt_embeds_mask  = prompt_embeds_mask,
    negative_prompt     = " ",
    true_cfg_scale      = 5.0,
    num_inference_steps = 50,
    height              = 1024,
    width               = 1024,
).images[0]

image.save("image_generation.png")