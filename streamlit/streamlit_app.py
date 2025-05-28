import streamlit as st
from datetime import datetime
import uuid
import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langgraph_pipeline import app
from core.utils import save_run_metadata

st.title("🧪 건강기능식품 평가 데모")

# 1. 이미지 업로드: 하나만 허용됨 (기본 설정)
uploaded_file = st.file_uploader("📷 제품 이미지 업로드", type=["png", "jpg", "jpeg"])

# 2. 썸네일 미리보기
if uploaded_file:
    image = Image.open(uploaded_file)
    image.thumbnail((300, 300))  # 썸네일 사이즈
    st.image(image, caption="업로드한 이미지 미리보기")

# 3. 사용자 질문 입력
user_query = st.text_input(
    "❓ 제품에 대해 궁금한 점을 입력하세요", "혈당에 도움이 되나요?"
)

# 4. 버튼을 눌러야 분석 시작
if st.button("🔍 분석 시작"):
    if not uploaded_file or not user_query:
        st.warning("이미지와 질문을 모두 입력해주세요.")
    else:
        with st.spinner("처리 중..."):
            # 5. 이미지 저장 (버튼 클릭 시 실행)
            os.makedirs("img", exist_ok=True)
            ext = os.path.splitext(uploaded_file.name)[-1].lower()
            unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
            test_image_path = os.path.join("img", unique_filename)
            with open(test_image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # 6. 파이프라인 실행
            run_id = (
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
            pipeline_start_time = datetime.now()

            initial_state = {
                "image_path": test_image_path,
                "user_query": user_query,
                "run_id": run_id,
                "image_data": None,
                "product_name_from_image": None,
                "enriched_info": None,
                "refined_user_query": None,
                "is_query_ambiguous": None,
                "evaluation_result": None,
                "data_consistency_status": None,
                "validation_issues": None,
                "final_response": None,
                "current_step": None,
                "error_message": None,
            }

            final_state = app.invoke(initial_state)
            pipeline_end_time = datetime.now()

            save_run_metadata(
                run_id,
                initial_state,
                final_state,
                pipeline_start_time,
                pipeline_end_time,
            )

        # 7. 결과 표시
        st.subheader("📊 결과 요약")
        if final_state.get("error_message"):
            st.error(
                f"❌ 오류: {final_state['error_message']} (단계: {final_state.get('current_step')})"
            )
        else:
            st.success("✅ 파이프라인 성공적으로 완료됨")
            st.markdown(
                f"**제품명:** {final_state.get('product_name_from_image', '없음')}"
            )
            st.markdown(
                f"**최종 판단:** {final_state.get('evaluation_result', {}).get('최종_판단', '없음')}"
            )
            st.markdown(
                f"**자연어 응답:** {final_state.get('final_response', '생성되지 않음')}"
            )

        st.markdown("### 📋 전체 상태 출력")
        st.json(final_state)
