import os
import json
import re
from datetime import datetime

# 📦 코드 블록(JSON) 정제
def extract_json_string(text: str) -> str:
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def save_step_output(data: dict, step_name: str, folder: str = "STEP_OUTPUTS") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{step_name}.json"
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 저장 완료: {file_path}")

    