import os
import json


def index_documents(
):
    count = 0
    for filename in os.listdir("json_data_chunked"):
        with open(f"json_data_chunked/{filename}", "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                count += 1
                print(f"Indexed document {count}")
    return f"success - {count} documents indexed"

index_documents()