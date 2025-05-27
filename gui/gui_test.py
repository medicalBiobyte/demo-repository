import os
import json
import sys
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QLineEdit,
    QMessageBox,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QTextCursor
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from core.config import text_llm  # text_llm 가져오기
from core.state_types import GraphState
from core.utils import extract_json_string, save_step_output

# 기존 core 모듈 임포트
from core.text_extract_1 import extract_info_from_image
from core.intent_refiner_agent_2 import node_refine_user_intent
from core.web_search_3 import get_enriched_product_info
from core.claim_check_4 import get_product_evaluation
from core.rag_service_4_1 import run_rag_from_ingredients
from core.data_validator_agent_5 import node_validate_data_consistency
from core.answer_user_6 import generate_natural_response

load_dotenv()


def node_extract_image_info(state: GraphState) -> Dict[str, Any]:
    state["current_step"] = "extract_image_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")
    image_path = state["image_path"]

    if not os.path.exists(image_path):
        error_msg = f"이미지 파일을 찾을 수 없습니다: {image_path}"
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    image_data_output = extract_info_from_image(image_path)

    if (
        not image_data_output
        or "제품명" not in image_data_output
        or not image_data_output.get("제품명")
    ):
        error_msg = "이미지에서 유효한 정보를 추출하지 못했습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        if image_data_output:
            print(json.dumps(image_data_output, indent=2, ensure_ascii=False))
        return {"image_data": image_data_output, "error_message": error_msg}

    product_name = image_data_output.get("제품명", "").split("/")[0].strip()
    if not product_name:
        error_msg = "추출된 제품명이 유효하지 않습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"image_data": image_data_output, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공. 제품명: {product_name}")

    output = {
        "image_data": image_data_output,
        "product_name_from_image": product_name,
        "error_message": None,
    }
    save_step_output(output, "extract_image_info")  # 저장

    return output


def node_enrich_product_info(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "enrich_product_info"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    product_name = state["product_name_from_image"]
    if not product_name:
        error_msg = "정보 보감을 위한 제품명이 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    enriched_data = get_enriched_product_info(product_name)

    if not enriched_data or enriched_data.get("error"):
        error_msg = f"웹 정보를 보감하지 못했습니다. 메시지: {enriched_data.get('error', '알 수 없음') if enriched_data else '데이터 없음'}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"enriched_info": enriched_data, "error_message": error_msg}

    if state.get("image_data"):
        enriched_data["original_효력_주장"] = state["image_data"].get("효력_주장")

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "enriched_info": enriched_data,
        "error_message": None,
    }
    save_step_output(output, "enrich_product_info")  # 저장

    return {"enriched_info": enriched_data, "error_message": None}


def node_evaluate_product(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "evaluate_product"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    enriched_info = state["enriched_info"]
    original_user_query = state["user_query"]  # 사용자의 원본 질문
    # ❗ 내부 평가에는 정제된 질문을 사용
    query_for_evaluation = state.get("refined_user_query", original_user_query)

    if not enriched_info:
        error_msg = "제품 평가를 위한 정보(enriched_info)가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    print(
        f" 평가에 사용될 질문: '{query_for_evaluation}' (원본 질문: '{original_user_query}')"
    )
    # ❗ get_product_evaluation 호출 시 정제된 질문(query_for_evaluation)을 전달
    # ❗ 그리고 get_product_evaluation 함수가 반환하는 결과에 '사용자_질문' 필드로 original_user_query를 넣어주도록 수정 필요
    evaluation_data = get_product_evaluation(
        enriched_data=enriched_info,
        user_query=query_for_evaluation,  # 내부 처리용 질문
        original_user_query_for_display=original_user_query,  # 최종 표기용 원본 질문 전달
    )

    # RAG 보완 로직 (기존과 동일하게 query_for_evaluation 사용)
    if evaluation_data.get("최종_판단", "").startswith(
        "광고 주장의 근거가 부족합니다"
    ):  # 최종_판단 문자열 시작 부분으로 체크
        print(
            f"🔁 '{evaluation_data.get('제품명', '알수없음')}' 제품에 대한 초기 평가 결과 근거 부족, RAG 보완 검색 실행..."
        )
        # run_rag_from_ingredients 호출 시에도 정제된 질문(query_for_evaluation) 사용
        rag_result = run_rag_from_ingredients(enriched_info, query_for_evaluation)
        if rag_result and rag_result.get("성분_기반_평가"):
            evaluation_data["RAG_보완"] = rag_result
            print(f"🔄 RAG 보완 결과: {rag_result.get('최종_판단')}")
        else:
            print(" RAG 보완 정보 없음 또는 유효하지 않음.")

    if not evaluation_data or "최종_판단" not in evaluation_data:
        error_msg = "제품 평가에 실패했습니다."
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"evaluation_result": evaluation_data, "error_message": error_msg}

    print(f"✅ {state['current_step']} 성공.")

    output = {
        "evaluation_result": evaluation_data,
        "error_message": None,
    }
    save_step_output(output, "evaluate_product")
    return output


def node_generate_natural_response(state: GraphState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    state["current_step"] = "generate_natural_response"
    print(f"--- 🏃 단계 실행: {state['current_step']} ---")

    evaluation_result = state["evaluation_result"]
    if not evaluation_result:
        error_msg = "답변 생성을 위한 평가 결과가 없습니다."
        print(f"❌ 오류: {error_msg}")
        return {"error_message": error_msg}

    print("🧠 LLM 응답 생성 요청 중...")
    response_text = generate_natural_response(evaluation_result)
    print("✅ LLM 응답 생성 완료")

    if not response_text or not isinstance(response_text, str):
        error_msg = f"답변 생성 실패 또는 문자열 반환 실패: {type(response_text)}"
        print(f"❌ {state['current_step']} 오류: {error_msg}")
        return {"final_response": response_text, "error_message": error_msg}

    print(
        f"✅ {state['current_step']} 성공. 응답: {response_text[:100]}..."
    )  # 앞 100자만 출력

    output = {
        "final_response": response_text,
        "error_message": None,
    }
    save_step_output(output, "generate_natural_response")  # 저장
    return output


class WorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    status = pyqtSignal(str)


class WorkerThread(QThread):
    def __init__(self, image_path, user_query):
        super().__init__()
        self.image_path = image_path
        self.user_query = user_query
        self.signals = WorkerSignals()
        self.workflow = self._create_workflow()
        self.is_running = True

    def _create_workflow(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("extract_image_info", node_extract_image_info)
        workflow.add_node("refine_user_intent", node_refine_user_intent)
        workflow.add_node("enrich_product_info", node_enrich_product_info)
        workflow.add_node("evaluate_product", node_evaluate_product)
        workflow.add_node("validate_data_consistency", node_validate_data_consistency)
        workflow.add_node("generate_response", node_generate_natural_response)

        workflow.set_entry_point("extract_image_info")
        workflow.add_edge("extract_image_info", "refine_user_intent")
        workflow.add_edge("refine_user_intent", "enrich_product_info")
        workflow.add_edge("enrich_product_info", "evaluate_product")
        workflow.add_edge("evaluate_product", "validate_data_consistency")
        workflow.add_edge("validate_data_consistency", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def run(self):
        try:
            initial_state = {
                "image_path": self.image_path,
                "user_query": self.user_query,
            }

            def progress_callback(step_name):
                if self.is_running:
                    print(f"📍 진행: {step_name}")  # <-- 로그 출력
                    self.signals.progress.emit(f"현재 단계: {step_name}")
                    self.signals.status.emit(f"분석 중... ({step_name})")

            final_state = self.workflow.invoke(initial_state)

            print(f"✅ 전체 워크플로우 완료. 상태: {final_state.keys()}")
            if self.is_running:
                self.signals.finished.emit(final_state)
                self.signals.status.emit("분석 완료")

        except Exception as e:
            import traceback

            traceback_str = traceback.format_exc()
            print(f"❌ 예외 발생: {traceback_str}")
            if self.is_running:
                self.signals.error.emit(f"{str(e)}\n\n{traceback_str}")
                self.signals.status.emit("오류 발생")

    def stop(self):
        self.is_running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker = None
        self.current_image_path = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()

    def initUI(self):
        self.setWindowTitle("제품 분석 시스템")
        self.setGeometry(100, 100, 1000, 800)

        # 메인 위젯과 레이아웃 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 이미지 선택 영역
        image_layout = QHBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """
        )
        image_layout.addWidget(self.image_label)

        # 이미지 선택 버튼
        select_image_btn = QPushButton("이미지 선택")
        select_image_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(select_image_btn)
        layout.addLayout(image_layout)

        # 사용자 질문 입력
        question_layout = QHBoxLayout()
        question_label = QLabel("질문:")
        question_label.setStyleSheet("font-size: 14px;")
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("제품에 대해 궁금한 점을 입력하세요")
        self.question_input.setStyleSheet(
            """
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
        """
        )
        self.question_input.returnPressed.connect(self.start_analysis)
        question_layout.addWidget(question_label)
        question_layout.addWidget(self.question_input)
        layout.addLayout(question_layout)

        # 분석 시작 버튼
        analyze_btn = QPushButton("분석 시작")
        analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )
        analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(analyze_btn)

        # 결과 출력 영역
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
        """
        )
        layout.addWidget(self.result_text)

        # 상태 표시줄
        self.statusBar().showMessage("준비됨")
        self.statusBar().setStyleSheet(
            """
            QStatusBar {
                background-color: #f0f0f0;
                color: #333;
                padding: 4px;
            }
        """
        )

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = file_name
            self.statusBar().showMessage(
                f"이미지 선택됨: {os.path.basename(file_name)}"
            )

    def start_analysis(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "이미지를 선택해주세요.")
            return

        user_query = self.question_input.text().strip()
        if not user_query:
            QMessageBox.warning(self, "경고", "질문을 입력해주세요.")
            return

        # 이전 워커가 있다면 정리
        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.statusBar().showMessage("분석 중...")

        # 사용자 질문 표시
        current_time = datetime.now().strftime("%H:%M:%S")
        self.result_text.append(
            f'<div style="margin: 10px 0;"><span style="color: #2196F3; font-weight: bold;">[{current_time}] 사용자:</span><br>{user_query}</div>'
        )
        self.result_text.append(
            '<div style="margin: 10px 0;"><span style="color: #4CAF50; font-weight: bold;">시스템:</span><br>분석을 시작합니다...</div>'
        )

        # 새로운 워커 생성 및 시작
        self.worker = WorkerThread(self.current_image_path, user_query)
        self.worker.signals.finished.connect(self.handle_results)
        self.worker.signals.error.connect(self.handle_error)
        self.worker.signals.progress.connect(self.handle_progress)
        self.worker.signals.status.connect(self.statusBar().showMessage)
        self.worker.start()

        # 질문 입력 필드 초기화
        self.question_input.clear()

        # 스크롤을 항상 아래로
        self.result_text.moveCursor(QTextCursor.End)

    def handle_progress(self, message):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.result_text.append(
            f'<div style="margin: 10px 0;"><span style="color: #9C27B0; font-weight: bold;">[{current_time}] 진행상황:</span><br>{message}</div>'
        )
        self.result_text.moveCursor(QTextCursor.End)

    def handle_results(self, final_state):
        if not self.worker or not self.worker.is_running:
            return

        current_time = datetime.now().strftime("%H:%M:%S")

        if final_state.get("error_message"):
            self.result_text.append(
                f'<div style="margin: 10px 0;"><span style="color: #f44336; font-weight: bold;">[{current_time}] 시스템:</span><br>❌ 오류 발생: {final_state["error_message"]}<br>오류 발생 단계: {final_state.get("current_step", "알 수 없음")}</div>'
            )
        elif final_state.get("final_response"):
            self.result_text.append(
                f'<div style="margin: 10px 0;"><span style="color: #4CAF50; font-weight: bold;">[{current_time}] 시스템:</span><br>✅ 분석이 완료되었습니다!<br><br>--- 최종 분석 결과 ---<br>{final_state["final_response"]}</div>'
            )
        else:
            self.result_text.append(
                f'<div style="margin: 10px 0;"><span style="color: #ff9800; font-weight: bold;">[{current_time}] 시스템:</span><br>⚠️ 분석은 완료되었지만, 결과를 생성하지 못했습니다.</div>'
            )

        self.statusBar().showMessage("분석 완료")
        self.result_text.append(
            '<div style="border-bottom: 1px solid #ddd; margin: 10px 0;"></div>'
        )

        # 스크롤을 항상 아래로
        self.result_text.moveCursor(QTextCursor.End)

    def handle_error(self, error_msg):
        if not self.worker or not self.worker.is_running:
            return

        current_time = datetime.now().strftime("%H:%M:%S")
        self.result_text.append(
            f'<div style="margin: 10px 0;"><span style="color: #f44336; font-weight: bold;">[{current_time}] 시스템:</span><br>❌ 오류 발생: {error_msg}</div>'
        )
        self.statusBar().showMessage("오류 발생")
        self.result_text.append(
            '<div style="border-bottom: 1px solid #ddd; margin: 10px 0;"></div>'
        )

        # 스크롤을 항상 아래로
        self.result_text.moveCursor(QTextCursor.End)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 스타일 설정
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
