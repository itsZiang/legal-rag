"""
llm_chunking.py

Chunk large text files using Google Gemini API.
Requires environment variable GEMINI_API_KEY and the package google-generative-ai.
"""

import os
import json
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()  # this will read the .env file in your CWD

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "Set GEMINI_API_KEY environment variable before running this script."
    )
client = genai.Client(api_key=api_key)


def llm_chunk_text(text: str, model: str = "gemini-2.0-flash") -> list:
    """
    Use Gemini LLM to split text into chunks of a specified number of tokens.
    """
    prompt = f"""
    Split the following legal article into chunks by grouping its numbered points (e.g., 1., 2., 3., etc.). Try to group these points into chunks with an estimated maximum of 1500 tokens and a minimum of 500 tokens. Do not split a point in half—keep each point intact when chunking. The token count does not need to be exact. Also, keep all numbering and formatting as-is, as this is legal text and those elements are important. Output the result as a list of chunks. Do not write python function, just generate the list directly using python format list of strings. 
    
    \n\n{text}
    """

    response = client.models.generate_content_stream(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            # max_output_tokens=1500,
            temperature=0.0
        ),
    )
    return response


output_dir = "../output_llm_chunking"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0
for filename in os.listdir("../output_long_context_5000"):
    if filename.endswith(".json"):
        print(f"Processing {filename}... NUMBER========================{count}")
        json_path = os.path.join("../output_long_context_5000", filename)
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        new_chunks = []
        for chunk in chunks:
            text = chunk["context"]
            title = chunk["title"]
            response = llm_chunk_text(text)
            final_text = ""
            for chunk in response:
                print(chunk.text, end="")
                final_text += chunk.text
            final_text = final_text.replace("```", "")
            final_text = final_text.replace("python", "")
            final_text = final_text.replace("json", "")
            final_text = final_text.replace("```", "")
            # right_side = final_text.split('=')[1].strip()
            chunks_list = eval(final_text)  # ["a", "b", "c"]
            for chunk in chunks_list:
                new_chunks.append({"title": title, "context": chunk})
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            json.dump(new_chunks, f, ensure_ascii=False, indent=2)
        print(f"Processed {filename}")
        count += 1
