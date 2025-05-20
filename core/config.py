import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from tavily import TavilyClient

# 🔐 .env 파일에서 환경변수 불러오기
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM 및 벡터 임베딩 설정
text_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

image_llm = genai.GenerativeModel("gemini-1.5-flash")
web_search_llm = TavilyClient(api_key=TAVILY_API_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

if __name__ == "__main__":
    test_prompt = "파이썬이란 무엇인가요?"

    # LangSmith 추적 구성
    config = RunnableConfig(configurable={"session_name": "graph_medi_test"})

    # LLM 호출 (수정됨)
    response = text_llm.invoke(test_prompt, config=config)

    # 결과 출력
    print("\n=== LLM 테스트 ===")
    print(f"질문: {test_prompt}")
    print(f"응답: {response.content}")
