import re
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# 📦 코드 블록(JSON) 정제
def extract_json_string(text: str) -> str:
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

STEP_OUTPUTS_DIR = "STEP_OUTPUTS"  # 기존 저장 폴더

def save_step_output(
    run_id: str,
    step_name: str,
    step_inputs: Optional[Dict[str, Any]],  # 단계 입력 값
    step_outputs: Dict[str, Any],         # 단계 반환 값 (기존 output)
    status: str = "success",              # 단계 실행 상태
    error_message: Optional[str] = None   # 오류 메시지
):
    """각 파이프라인 단계의 입력, 출력 및 상태를 run_id별 폴더에 저장합니다."""
    if not run_id:
        print("⚠️ 경고: run_id가 제공되지 않아 단계 출력을 저장할 수 없습니다.")
        return

    run_dir = os.path.join(STEP_OUTPUTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 파일명은 단계 이름으로 단순화 (타임스탬프는 내용에 포함)
    filename = f"{step_name}.json"
    filepath = os.path.join(run_dir, filename)

    data_to_save = {
        "run_id": run_id,
        "step_name": step_name,
        "timestamp_iso": datetime.now().isoformat(), # 저장 시점의 타임스탬프
        "status": status,
        "inputs": step_inputs if step_inputs else {},
        "outputs": step_outputs, # 기존에 output으로 저장하던 내용
        "error_message": error_message,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"💾 [{run_id}] 단계 출력 저장 완료: {filepath}")
    except Exception as e:
        print(f"❌ [{run_id}] '{filepath}' 저장 실패: {e}")

def save_run_metadata(
    run_id: str,
    initial_state: Dict[str, Any],
    final_state: Dict[str, Any],
    pipeline_start_time: datetime,
    pipeline_end_time: datetime
):
    """파이프라인 전체 실행에 대한 메타데이터를 저장합니다."""
    if not run_id:
        print("⚠️ 경고: run_id가 제공되지 않아 실행 메타데이터를 저장할 수 없습니다.")
        return

    run_dir = os.path.join(STEP_OUTPUTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    filepath = os.path.join(run_dir, "_run_metadata.json") # 메타데이터 파일명

    run_duration_seconds = (pipeline_end_time - pipeline_start_time).total_seconds()
    
    overall_status = "success"
    if final_state.get("error_message"):
        overall_status = "failure"
    elif not final_state.get("final_response"): # 또는 성공을 판단하는 다른 기준
        overall_status = "completed_with_warnings"


    metadata = {
        "run_id": run_id,
        "pipeline_start_time_iso": pipeline_start_time.isoformat(),
        "pipeline_end_time_iso": pipeline_end_time.isoformat(),
        "pipeline_duration_seconds": run_duration_seconds,
        "overall_status": overall_status,
        "initial_inputs": {
            "image_path": initial_state.get("image_path"),
            "user_query": initial_state.get("user_query"),
        },
        "final_main_output": final_state.get("final_response") if overall_status == "success" else None,
        "final_error_message": final_state.get("error_message"),
        "last_step_executed": final_state.get("current_step"),
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"📜 [{run_id}] 실행 메타데이터 저장 완료: {filepath}")
    except Exception as e:
        print(f"❌ [{run_id}] 메타데이터 저장 실패: {e}")