import json
import os

INPUT_PATH = "/home/hejiawei/data_preprocess/fragmol/frag_train.jsonl"


def count_records(jsonl_path):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            if line.strip() == "":
                if buffer:
                    try:
                        json.loads(buffer)
                        count += 1
                    except json.JSONDecodeError:
                        pass
                    buffer = ""
            else:
                buffer += line

        if buffer:
            try:
                json.loads(buffer)
                count += 1
            except json.JSONDecodeError:
                pass

    return count


if __name__ == "__main__":
    total = count_records(INPUT_PATH)
    print(f"Total records: {total}")
