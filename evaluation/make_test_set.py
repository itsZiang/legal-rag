import json

def main():
    with open("123.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip().split(";")
        if len(line) != 2:
            continue
        question = line[0]
        doc_ids = line[1].strip().split(",")
        doc_ids = [int(x.strip()) for x in doc_ids if x.strip()]
        data.append({"question": question.strip(), "doc_ids": doc_ids})

    with open("test_set.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()