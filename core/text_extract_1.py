import os
import json
import re
from PIL import Image
from dotenv import load_dotenv
from .config import image_llm # 앞에 . 을 추가합니다.
from .prompt import IMG2TEXT_PROMPT # 앞에 . 을 추가합니다.

# 🔐 환경변수 로드
load_dotenv()

IMG_DIR = "img"
OUTPUT_PATH = "IMG2TEXT_data/result_all.json"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# 🖼️ 이미지 로드
def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


# 📦 코드 블록(JSON) 정제
def extract_json_string(text: str) -> str:
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# 📤 LLM 호출 및 결과 추출
def extract_info_from_image(image_path: str) -> dict:
    image = load_image(image_path)
    prompt_text = IMG2TEXT_PROMPT
    response = image_llm.generate_content([prompt_text, image])
    raw_text = response.text
    cleaned = extract_json_string(raw_text)

    try:
        data = json.loads(cleaned)
        data["이미지"] = os.path.basename(image_path)  # 파일명 포함
        return data
    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패: {os.path.basename(image_path)}")
        print("응답:\n", raw_text)
        return {}


# 🚀 전체 이미지 처리 후 하나로 저장
def process_all_images():
    all_results = []

    for filename in os.listdir(IMG_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        image_path = os.path.join(IMG_DIR, filename)
        print(f"🔍 처리 중: {filename}")
        result = extract_info_from_image(image_path)

        if result:
            all_results.append(result)
            print(f"✅ 추출 완료: {filename}")
        else:
            print(f"⚠️ 실패: {filename}")

    # 모든 결과를 하나의 JSON 배열로 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n📦 전체 결과 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    process_all_images()
