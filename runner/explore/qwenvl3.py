import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

# qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen3-VL-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen3-VL-8B-Instruct")
qwenvl = AutoModelForImageTextToText.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
# tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/Qwen/Qwen-Image/text_encoder")
device = torch.device("cuda:0")
qwenvl = qwenvl.to(device)
qwenvl.eval()

# magic_prompt = "Ultra HD, 4K, cinematic composition"


SYSTEM_PROMPT = '''
You are a professional Prompt Optimizer specializing in image generation models (e.g., MidJourney, Stable Diffusion), with deep expertise in the visualization logic of such models. Your core task is to rewrite user-provided prompts into highly clear, easy-to-render versions that align with the model's understanding habits.
If the prompt describes the result of a process (e.g., "a cup after being poured with hot water"), use your background knowledge (e.g. scientific facts, cultural common sense, and logical reasoning) to reasonably infer this result. The inference must be based on objective logic and avoid subjective imagination beyond common cognition.
Focus strictly on describing the final visual appearance of the scene. Clarify key elements of the main subject, including but not limited to its shape, color, state, texture, proportion, and interaction with the surrounding environment.
After receiving the user's prompt to be rewritten, first explain your optimization reasoning. This reasoning should include two parts: 1) the key issues of the original prompt (e.g., vague state description, missing color information); 2) the purpose of each improvement (e.g., adding texture details to help the model render realism). Then, output the final revised prompt in the fixed format of "Revised Prompt: {}", where the specific revised content is filled in the "{}".
'''

json_path = "/data/phd/jinjiachun/codebase/WISE/data"
json_file_names = ["cultural_common_sense.json", "natural_science.json", "spatio-temporal_reasoning.json"]

# 创建统一的输出文件
output_path = os.path.join(json_path, "qwenvl_new_system_prompt.jsonl")

# 打开输出文件进行写入
with open(output_path, "w", encoding="utf-8") as output_f:
    for json_file_name in json_file_names:
        with open(os.path.join(json_path, json_file_name), "r") as f:
            data = json.load(f)
            
            for item in data:
                prompt = item["Prompt"]
                prompt_id = item["prompt_id"]
                # print(prompt_id, prompt)

                original_prompt = prompt

                original_prompt = original_prompt.strip()
                prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\n Revised Prompt:"
                prompt = [prompt]
                template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

                txt = [template.format(e) for e in prompt]

                txt_tokens = tokenizer(
                    txt, max_length=10240, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                generation_output = qwenvl.generate(
                    **txt_tokens, 
                    max_new_tokens=512,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                generated_ids = generation_output.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(txt_tokens.input_ids, generated_ids)
                ]
                output_text = tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                output_text = output_text.replace("\n", " ")
                output_text = output_text
                
                # 创建要写入的数据
                output_data = {
                    "prompt_id": prompt_id,
                    "output_text": output_text
                }
                
                # 写入JSONL文件
                output_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                print(f"{prompt_id}: {output_text}")

print(f"所有数据已保存到: {output_path}")