import json
import os
import re


def save_to_json(chunks, output_file):
    """Save the processed data to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)


def extract_section_text(text):
    """
    Extract section text from the given text.
    """
    section_pattern = r"(Mục\s+\d+[a-z]*\.?\s*\n*\s*((?:[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ,;:\-\s]+)))"

    # Tìm tất cả các phần khớp
    matches = re.findall(section_pattern, text, flags=re.DOTALL)

    # Lấy phần toàn bộ chuỗi khớp (group 0) hoặc group 1 nếu muốn đầy đủ hơn
    section_text = [match[0].strip() for match in matches]
    for i in range(len(section_text)):
        if section_text[i].endswith("\n\nĐ"):
            section_text[i] = section_text[i][:-3]
    # print(section_text)
    return section_text


def extract_legal_name(text):
    """
    Extract legal name from the given text.
    """
    match = re.search(r"\d+\/[A-Z]*[^\s]*", text)
    return match.group()


def preprocess_and_chunk_text(text):
    """
    Preprocess the text and split it into chunks with titles and contexts.
    - Titles include both the current chapter and article.
    - Contexts contain the text under each article.
    """

    # Define regex patterns for identifying chapters and articles
    chapter_pattern = r"(Chương\s+[IVXLCDM]+\s*[\n\r]*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰÝ][^\n]*)"
    article_pattern = r"(Điều\s+\d+[a-z]?\.\s*[\n\r]*[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ][^\n]*[\n\r]*[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự]*[^\dA-Z]*)"
    section_text_list = extract_section_text(text)
    legal_name = extract_legal_name(text)

    # Split text into parts using chapter or article
    sections = re.split(f"({chapter_pattern}|{article_pattern})", text, flags=re.DOTALL)

    current_chapter = None
    current_article = None
    chunks = []
    buffer = ""

    for section in sections:
        if section is None or section.strip() == "":
            continue

        section = section.strip()

        if re.match(chapter_pattern, section):
            if current_article and buffer.strip():
                for section_text in section_text_list:
                    if section_text in buffer:
                        buffer = buffer.replace(section_text, "")
                chunks.append(
                    {
                        "title": f"{current_article} {current_chapter or ''}, {legal_name}".strip(),
                        "context": buffer.strip().split("./.")[0],
                    }
                )
            current_chapter = section
            current_article = None
            buffer = ""
            continue

        if re.match(article_pattern, section):
            if current_article and buffer.strip():
                for section_text in section_text_list:
                    if section_text in buffer:
                        buffer = buffer.replace(section_text, "")
                chunks.append(
                    {
                        "title": f"{current_article} {current_chapter or ''}, {legal_name}".strip(),
                        "context": buffer.strip().split("./.")[0],
                    }
                )
            current_article = section
            buffer = ""
            continue

        if current_article:
            buffer += " " + section.strip()

    # Nếu còn phần cuối
    if current_article and buffer.strip():
        chunks.append(
            {
                "title": f"{current_article} {current_chapter or ''}, {legal_name}".strip(),
                "context": buffer.strip().split("./.")[0],  # thêm xử lý split ở đây
            }
        )

    return chunks


# Đường dẫn thư mục
txt_folder = "../txt_data"
output_folder = "../output"

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua tất cả file .txt trong thư mục txt
for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        # Đường dẫn file txt
        txt_path = os.path.join(txt_folder, filename)

        # Đọc nội dung file
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Xử lý text
        chunks = preprocess_and_chunk_text(text)
        legal_name = extract_legal_name(text)

        # In thông tin
        print(f"File: {filename}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Legal number: {legal_name}")

        # Tạo tên file đầu ra (thay .txt thành .jsonl)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)

        # Lưu vào file .jsonl
        save_to_json(chunks, output_path)
        print(f"Saved to: {output_path}\n")
        print("=========================")
