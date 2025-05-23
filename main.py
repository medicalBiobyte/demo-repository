import os
import sys
import argparse
import json
import importlib.util
from dotenv import load_dotenv
from core.config import logger
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# 각 모듈 불러오기
from core.text_extract_1 import process_all_images
from core.web_search_3 import process_all_products
from core.claim_check_4 import evaluate_product
from core.answer_user_6 import generate_natural_response

# 디렉터리 상수 정의
IMG_DIR = "img"
IMG2TEXT_DIR = "IMG2TEXT_data"
TEXT2SEARCH_DIR = "TEXT2SEARCH_data"
DECISION_DIR = "DECISION_data"


# 에이전트 추상 클래스
class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        logger.info(f"에이전트 '{name}' 초기화됨")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    def __str__(self) -> str:
        return f"Agent({self.name})"


# 이미지 텍스트 추출 에이전트
class ImageTextExtractAgent(Agent):
    def __init__(self):
        super().__init__("이미지 텍스트 추출")

    def process(self, input_data: Any = None) -> List[Dict]:
        logger.info(f"{self.name} 작업 시작")
        text_extract.process_all_images()

        # 결과 반환
        result_path = "IMG2TEXT_data/result_all.json"
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


# 웹 검색 에이전트
class WebSearchAgent(Agent):
    def __init__(self):
        super().__init__("웹 검색 및 성분 추출")

    def process(self, input_data: List[Dict]) -> List[str]:
        logger.info(f"{self.name} 작업 시작")
        web_search.process_all_products()

        # 처리된 파일 목록 반환
        processed_files = []
        for filename in os.listdir(TEXT2SEARCH_DIR):
            if filename.endswith(".json"):
                processed_files.append(os.path.join(TEXT2SEARCH_DIR, filename))

        return processed_files


# 검증 에이전트
class ClaimCheckAgent(Agent):
    def __init__(self, query: str):
        super().__init__("광고 주장 검증")
        self.query = query

    def process(self, input_data: List[str]) -> List[str]:
        logger.info(f"{self.name} 작업 시작: 질문 - '{self.query}'")

        processed_files = []
        for file_path in input_data:
            if os.path.exists(file_path):
                claim_check.evaluate_product(file_path, self.query)

                # 출력 파일 이름 구성
                base_name = os.path.basename(file_path)
                if base_name.startswith("enriched_"):
                    product_name = base_name[9:-5]  # 'enriched_'와 '.json' 제거
                else:
                    product_name = base_name[:-5]  # '.json' 제거

                processed_files.append(
                    os.path.join(DECISION_DIR, f"enriched_{product_name}.json")
                )

        return processed_files


# 답변 생성 에이전트
class AnswerGenerationAgent(Agent):
    def __init__(self):
        super().__init__("답변 생성")

    def process(self, input_data: List[str]) -> List[str]:
        logger.info(f"{self.name} 작업 시작")

        answers = []
        for file_path in input_data:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    answer = answer_user.generate_natural_response(data)
                    answers.append(answer)

                    print("\n" + "=" * 60)
                    print(answer)
                    print("=" * 60)

        return answers


# 파이프라인 클래스
class Pipeline:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        logger.info(f"파이프라인 초기화: {len(agents)}개 에이전트")

    def run(self, initial_input: Any = None) -> Any:
        result = initial_input

        for i, agent in enumerate(self.agents, 1):
            logger.info(f"파이프라인 단계 {i}/{len(self.agents)}: {agent.name}")
            result = agent.process(result)

        return result


# 필요한 디렉터리 생성
def create_directories():
    for dir_path in [IMG_DIR, IMG2TEXT_DIR, TEXT2SEARCH_DIR, DECISION_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"디렉터리 생성/확인: {dir_path}")


def main():
    # 환경 변수 로드
    load_dotenv()

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="의약품 광고 검증 파이프라인")
    parser.add_argument(
        "--query",
        type=str,
        default="이 약은 키 크는데 도움이 되나요?",
        help="사용자 질문 (예: '이 약은 키 크는데 도움이 되나요?')",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="시작할 단계 (1: 이미지 텍스트 추출, 2: 웹 검색, 3: 검증, 4: 답변 생성, 0: 전체)",
    )
    args = parser.parse_args()

    # 디렉터리 생성
    create_directories()

    # 에이전트 준비
    agents = [
        ImageTextExtractAgent(),
        WebSearchAgent(),
        ClaimCheckAgent(args.query),
        AnswerGenerationAgent(),
    ]

    # 시작 단계에 따라 에이전트 필터링
    if args.step > 0:
        agents = agents[args.step - 1 :]
        logger.info(f"{args.step}단계부터 시작: {agents[0].name}")

    # 파이프라인 실행
    pipeline = Pipeline(agents)
    result = pipeline.run()

    logger.info("파이프라인 실행 완료")
    return result


if __name__ == "__main__":
    main()
