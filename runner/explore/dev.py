# load "/Users/orres/Playground/qimage/data/culture.jsonl"
import os
import json

path = "/Users/orres/Playground/qimage/data"
json_file_names = ["culture.jsonl", "spatio.jsonl", "science.jsonl"]

idx = 1

for json_file_name in json_file_names:
    file_path = os.path.join(path, json_file_name)
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            response = data["response"]["body"]["choices"][0]["message"]["content"]
            response = response.split("{")[1].split("}")[0]
            print(idx, response)
            idx += 1
