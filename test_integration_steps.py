import os
import json
from dotenv import load_dotenv
from core.text_extract_1 import extract_info_from_image 
from core.web_search_2 import get_enriched_product_info
from core.claim_check_3 import get_product_evaluation
from core.answer_user_4 import generate_natural_response

load_dotenv()

# --- 테스트용 입력값 설정 ---
# 실제 테스트할 이미지 파일 경로 
TEST_IMAGE_DIR = "img"

test_image_filename = "height_medi_1.png" # 👈 실제 테스트 이미지 파일명으로 변경!
test_image_path = os.path.join(TEST_IMAGE_DIR, test_image_filename)

# 예시 사용자 질문
sample_user_query = "이거 먹으면 키 크는데 효과 있나요?"

# 중간 결과를 저장하거나 확인하기 위한 함수 
def print_and_save_json(data, filename):
    print(f"\n--- {filename} 내용 ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    # 필요하다면 파일로 저장
    # with open(filename, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=2, ensure_ascii=False)
    print("-" * 30)

def run_integration_tests():
    print("🚀 통합 테스트 시작 🚀")

    # === 1단계: 이미지에서 정보 추출 테스트 ===
    print("\n[테스트 1/4] 이미지 정보 추출 (extract_info_from_image)")
    if not os.path.exists(test_image_path):
        print(f"❌ 테스트 이미지 파일을 찾을 수 없습니다: {test_image_path}")
        print(f"'{TEST_IMAGE_DIR}' 폴더에 '{test_image_filename}' 파일을 준비해주세요.")
        return

    image_data = extract_info_from_image(test_image_path)
    if not image_data or "제품명" not in image_data or not image_data.get("제품명"):
        print("❌ 1단계 실패: 이미지에서 유효한 정보를 추출하지 못했습니다.")
        print_and_save_json(image_data, "step1_image_data_error.json")
        return
    print("✅ 1단계 성공: 이미지 정보 추출 완료.")
    print_and_save_json(image_data, "step1_image_data.json")
    
    product_name_from_image = image_data.get("제품명", "").split("/")[0].strip()
    if not product_name_from_image:
        print("❌ 1단계 실패: 추출된 제품명이 유효하지 않습니다.")
        return


    # === 2단계: 웹 검색으로 제품 정보 보강 테스트 ===
    print("\n[테스트 2/4] 웹 정보 보강 (get_enriched_product_info)")
    # 입력: 1단계에서 얻은 제품명
    enriched_info = get_enriched_product_info(product_name_from_image)
    if not enriched_info or enriched_info.get("error"):
        print(f"❌ 2단계 실패: 웹 정보를 보강하지 못했습니다. 메시지: {enriched_info.get('error', '알 수 없음')}")
        print_and_save_json(enriched_info, "step2_enriched_info_error.json")
        # 웹 정보 보강에 실패했더라도, 다음 단계를 위해 최소한의 정보는 유지할 수 있습니다.
        # 여기서는 테스트이므로, 심각한 오류로 간주하고 중단하거나, 또는 경고 후 진행할 수 있습니다.
        # enriched_info = {"제품명": product_name_from_image, "확정_성분": [], "요약_텍스트": "웹 정보 조회 실패"} # 예시: 최소 정보
        return # 오류 시 중단
        
    print("✅ 2단계 성공: 웹 정보 보강 완료.")
    print_and_save_json(enriched_info, "step2_enriched_info.json")
    
    # 다음 단계를 위해 1단계의 원본 효능 주장을 enriched_info에 추가해줄 수 있습니다 (선택적).
    # 이는 get_product_evaluation 함수가 이 정보를 기대하는 경우 유용합니다.
    enriched_info["original_효능_주장"] = image_data.get("효능_주장")


    # === 3단계: 사용자 질문 기반 제품 평가 테스트 ===
    print("\n[테스트 3/4] 제품 평가 (get_product_evaluation)")
    # 입력: 2단계에서 얻은 보강된 정보, 사용자 질문
    evaluation_result = get_product_evaluation(enriched_info, sample_user_query)
    if not evaluation_result or "최종_판단" not in evaluation_result:
        print("❌ 3단계 실패: 제품 평가에 실패했습니다.")
        print_and_save_json(evaluation_result, "step3_evaluation_result_error.json")
        return
    print("✅ 3단계 성공: 제품 평가 완료.")
    print_and_save_json(evaluation_result, "step3_evaluation_result.json")


    # === 4단계: 자연어 답변 생성 테스트 ===
    print("\n[테스트 4/4] 자연어 답변 생성 (generate_natural_response)")
    # 입력: 3단계에서 얻은 평가 결과
    natural_response = generate_natural_response(evaluation_result)
    if not natural_response or not isinstance(natural_response, str):
        print("❌ 4단계 실패: 자연어 답변 생성에 실패했거나 결과가 문자열이 아닙니다.")
        print(f"반환된 값: {natural_response}")
        return
    print("✅ 4단계 성공: 자연어 답변 생성 완료.")
    print("\n--- 최종 생성된 자연어 답변 ---")
    print(natural_response)
    print("-" * 30)

    print("\n🎉 모든 통합 테스트 단계가 성공적으로 완료되었습니다! (오류 없이 실행된 경우)")

if __name__ == "__main__":
    # 테스트 실행 전, test_image_filename이 실제 파일인지 확인
    if test_image_filename == "sample_ad_image.jpg": # 사용자가 기본값을 변경했는지 확인
        print("="*50)
        print(f"🚨 경고: `test_image_filename` 변수를 '{TEST_IMAGE_DIR}' 폴더에 있는 실제 테스트 이미지 파일명으로 변경해주세요!")
        print(f"현재 설정: {test_image_filename}")
        print("="*50)
    elif not os.path.exists(os.path.join(TEST_IMAGE_DIR, test_image_filename)):
         print("="*50)
         print(f"🚨 오류: 테스트 이미지 파일 '{os.path.join(TEST_IMAGE_DIR, test_image_filename)}'을 찾을 수 없습니다.")
         print(f"'{TEST_IMAGE_DIR}' 폴더를 확인하고, `test_image_filename` 변수를 올바르게 설정해주세요.")
         print("="*50)
    else:
        run_integration_tests()