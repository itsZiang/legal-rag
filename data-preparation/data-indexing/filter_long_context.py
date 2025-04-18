import json
import os

json_dir = "../output"


def save_to_json(chunks, output_file):
    """Save the processed data to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)


if os.path.exists("output_long_context_2048"):
    os.remove("output_long_context_2048")

if os.path.exists("output_long_context_4096"):
    os.remove("output_long_context_4096")

if os.path.exists("output_long_context_8192"):
    os.remove("output_long_context_8192")

sizes = [2048, 4096, 5000]


for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        # Đường dẫn file json
        json_path = os.path.join(json_dir, filename)
        # Đọc nội dung file
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for size in sizes:
            count = 0
            new_chunks = []
            output_dir = f"../output_long_context_{size}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, filename)
            for chunk in chunks:
                context = chunk["context"]
                title = chunk["title"]

                if len(context) > size:
                    count += 1
                    new_chunks.append({"title": title, "context": context})
            if count > 0:
                save_to_json(new_chunks, output_path)

num_2048 = 0
num_4096 = 0
num_5000 = 0

for filename in os.listdir("../output_long_context_2048"):
    if filename.endswith(".json"):
        json_path = os.path.join("../output_long_context_2048", filename)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks_2048 = json.load(f)
            num_2048 += len(chunks_2048)

for filename in os.listdir("../output_long_context_4096"):
    if filename.endswith(".json"):
        json_path = os.path.join("../output_long_context_4096", filename)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks_4096 = json.load(f)
            num_4096 += len(chunks_4096)


for filename in os.listdir("../output_long_context_5000"):
    if filename.endswith(".json"):
        json_path = os.path.join("../output_long_context_5000", filename)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks_5000 = json.load(f)
            num_5000 += len(chunks_5000)

print(f"2048: {num_2048}")
print(f"4096: {num_4096}")
print(f"5000: {num_5000}")

