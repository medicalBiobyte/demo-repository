import os
import json

DECISION_DIR = "DECISION_data"


def generate_natural_response(data: dict) -> str:
    product = data.get("제품명", "알 수 없음")
    query = data.get("사용자_질문", "")
    keywords = data.get("질문_핵심_키워드", [])
    match_info = data.get("매칭_성분", [])
    fallback = data.get("제품명_기반_보완", {})
    final_decision = data.get("최종_판단", "")

    response_lines = [f"🔍 **{product}** 제품에 대한 평가 결과입니다."]
    response_lines.append(f'사용자 질문: "{query}"')
    if keywords:
        response_lines.append(f"▶️ 질문에서 추출된 핵심 키워드: {', '.join(keywords)}")

    # 성분 기반 평가
    response_lines.append("\n📌 **성분별 효능 평가:**")
    for entry in match_info:
        name = entry["성분명"]
        efficacy = entry["효능"]
        match = entry["일치도"]

        if efficacy == "없음" or match == "정보 없음":
            response_lines.append(f"- {name}: ⚠️ 공공 데이터에 정보 없음")
        else:
            icon = "✅" if match == "일치" else "❌"
            response_lines.append(f'- {name}: "{efficacy}" ({icon} {match})')

    # 제품명 기반 보완 설명 (있을 경우)
    if fallback and fallback.get("보완_효능"):
        response_lines.append(
            "\n🔄 **성분 정보가 부족하여 제품명 기준으로 보완된 효능:**"
        )
        match_icon = "✅" if fallback.get("일치도") == "일치" else "❌"
        response_lines.append(
            f"- 제품 설명: \"{fallback['보완_효능']}\" ({match_icon} {fallback['일치도']})"
        )

    # 최종 판단
    response_lines.append("\n🧾 **최종 판단:** " + final_decision)

    return "\n".join(response_lines)


# 🧪 실행 예시: 모든 평가 결과 출력
if __name__ == "__main__":
    for filename in os.listdir(DECISION_DIR):
        if filename.endswith(".json"):
            path = os.path.join(DECISION_DIR, filename)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                answer = generate_natural_response(data)

                print("\n" + "=" * 60)
                print(answer)
                print("=" * 60)
