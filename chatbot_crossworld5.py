import streamlit as st
from langchain_openai import ChatOpenAI
import os
import bs4
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import PromptTemplate

# Streamlit UI 설정
st.set_page_config(page_title="제2의 나라 chatbot", page_icon=":video_game:")
st.title("💟 netmarble 💟")
st.header("✨ 제2의 나라: Cross Worlds ✨")
st.caption("😄 Jiyun Park 😄")


# 사이드바에 Upstage API 키 입력란 추가
with st.sidebar:
    upstage_api_key = st.text_input("Upstage API Key", key="chatbot_api_key", type="password")

# 사이드바에 Tavily API 키 입력란 추가
with st.sidebar:
    tavily_api_key = st.text_input("Tavily API Key", key="tavily_api_key", type="password")

# Upstage API 키 확인
if upstage_api_key:
    os.environ["UPSTAGE_API_KEY"] = upstage_api_key
else:
    st.info("Upstage API Key를 입력해주세요.")
    st.stop()

# Tavily API 키 확인
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
else:
    st.info("Tavily API Key를 입력해주세요.")
    st.stop()


# CSV 파일 경로 입력
import streamlit as st
import pandas as pd


#csv_file_path = "C:/Users/jyp/.conda/envs/langchain-cource/00_PT/total_with_images_fake.csv"
csv_file_path = "total_with_images.csv"
loader = CSVLoader(file_path=csv_file_path)
docs = loader.load()


# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# vectorstore & retriever
vectorstore = FAISS.from_documents(splits, UpstageEmbeddings(model="solar-embedding-1-large"))
retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "solar_search",
    "Searches any questions related to Solar. Always use this tool when user query is related to Solar!",
)

from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool = TavilySearchResults()

tools = [tavily_tool,retriever_tool]

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub


# prompt
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
    당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
    당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
    검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
    만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, tavily를 통해 제2의나라라는 키워드와 함께 검색을 실시하고, 만약
    검색을 통해 답을 얻었다면, 그에 대한 출처를 남겨주세요. 하지만 검색해도
    답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
    한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.
    그리고 당신을 사용하는 사람들을 넷마블의 "제2의 나라"라는 모바일 게임의 유저와 운영자입니다. 
    Don't narrate the answer, just answer the question. Let's think step-by-step.

    #Question: 
    {question} 

    #Context: 
    {context} 

    #Answer:
    """
)


# prompt = hub.pull("teddynote/rag-prompt-korean")
# prompt

import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate

# 대화 히스토리를 포함한 프롬프트 템플릿 정의
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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


agent_executor = AgentExecutor(agent=agent, tools=tools)



rag_chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "history": lambda x: x["history"],
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import asyncio

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

class StreamChain:
    def __init__(self, executor):
        self.executor = executor
    
    async def astream(self, query, history):
        stream_handler = StreamingStdOutCallbackHandler()
        response = await self.executor.arun({"input": query, "history": history}, callbacks=[stream_handler])
        return response

chain = StreamChain(agent_executor)

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
    
    # 대화 히스토리 생성
    history = "\n".join([f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in st.session_state.messages[1:-1]])
    
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        try:
            full_response = asyncio.run(chain.astream(prompt, history))
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            st.error(full_response)
    
    st.session_state.messages.append(AIMessage(content=full_response))

st.empty()
