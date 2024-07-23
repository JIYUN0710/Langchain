import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler

# Streamlit UI 설정
st.set_page_config(page_title="제2의 나라 chatbot", page_icon=":video_game:")
st.title("💟 netmarble 💟")
st.header("✨ 제2의 나라: Cross Worlds ✨")
st.caption("😄 Jiyun Park 😄")

# 사이드바에 API 키 입력란 추가
with st.sidebar:
    upstage_api_key = st.text_input("Upstage API Key", key="chatbot_api_key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", key="tavily_api_key", type="password")

# API 키 확인
if not upstage_api_key or not tavily_api_key:
    st.info("Upstage API Key와 Tavily API Key를 모두 입력해주세요.")
    st.stop()

os.environ["UPSTAGE_API_KEY"] = upstage_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

# CSV 파일 로드 및 처리
@st.cache_resource
def load_data():
    csv_file_path = "total_with_images.csv"
    loader = CSVLoader(file_path=csv_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, UpstageEmbeddings(model="solar-embedding-1-large"))
    return vectorstore

vectorstore = load_data()
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "solar_search",
    "Searches any questions related to Solar. Always use this tool when user query is related to Solar!",
)

tavily_tool = TavilySearchResults()

tools = [tavily_tool, retriever_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for the mobile game '제2의 나라' by Netmarble. 
    당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
    당신의 임무는 주어진 문맥(context)에서 주어진 질문(question)에 답하는 것입니다. 
    검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요.

    만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, tavily를 통해 제2의나라라는 키워드와 함께 검색을 실시하고, 
    만약 검색을 통해 답을 얻었다면, 그에 대한 출처를 남겨주세요. 
    하지만 검색해도 답을 모른다면 '주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다'라고 답하세요.

    한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.
    그리고 당신을 사용하는 사람들은 넷마블의 '제2의 나라'라는 모바일 게임의 유저와 운영자입니다.

    Use the following context to answer the user's question: {context}
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatUpstage()
agent = create_tool_calling_agent(llm, tools, prompt)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant for the mobile game '제2의 나라' by Netmarble.")
    ]

for message in st.session_state.messages[1:]:  # 시스템 메시지 제외
    with st.chat_message(message.type):
        st.markdown(message.content)

prompt = st.chat_input("제2의 나라에 대해 무엇을 알고 싶으신가요?")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)
    
    history = "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in st.session_state.messages[1:-1]])
    
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        try:
            response = agent_executor.invoke(
                {"input": prompt},
                config={"callbacks": [stream_handler]}
            )
            st.markdown(stream_handler.text)
            st.session_state.messages.append(AIMessage(content=stream_handler.text))
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))

st.empty()