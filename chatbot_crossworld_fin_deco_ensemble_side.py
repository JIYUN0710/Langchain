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
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Streamlit UI 설정
#st.set_page_config(page_title="제2의 나라 chat", page_icon=":video_game:")
#st.header("제2의 나라 chat *^^*")
# Streamlit UI 설정


import streamlit as st

# CSS를 사용하여 사이드바 배경 이미지 및 텍스트 스타일 설정
sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-image: url("https://raw.githubusercontent.com/JIYUN0710/Langchain/9d34069783f1e72e837cee1333c95a03786732c8/crossworld_2.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(0, 0, 0, 0.5);  /* 반투명 검은색 오버레이 */
    padding: 20px;
}
[data-testid="stSidebar"] .stTextInput label, 
[data-testid="stSidebar"] .stSelectbox label {
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
[data-testid="stSidebar"] .stTextInput input, 
[data-testid="stSidebar"] .stSelectbox select {
    color: black !important;
    background-color: rgba(255, 255, 255, 0.8) !important;
    font-size: 16px !important;
}
</style>
"""

# Streamlit UI 설정
st.set_page_config(page_title="제2의 나라 chatbot", page_icon=":video_game:")
st.title("💟 netmarble 💟")
st.header("✨ 제2의 나라: Cross Worlds ✨")
st.caption("😄 Jiyun Park 😄")

# 사이드바 스타일 적용
st.markdown(sidebar_style, unsafe_allow_html=True)

# 사이드바 내용
with st.sidebar:
    upstage_api_key = st.text_input("Upstage API Key", key="chatbot_api_key", type="password")
    people = st.selectbox('당신은 제2의 나라 User 인가요?',
                          ('YES', 'NO'),
                          index=0)
    if people == 'YES':
        email = st.text_input('당신의 게임 이메일을 입력해주세요')
    elif people == 'NO':
        num = st.text_input('당신의 사원번호를 입력해주세요')

# API 키 확인 및 메시지 표시
if not upstage_api_key:
    st.warning("Upstage API Key를 입력해주세요.")
    st.stop()
else:
    os.environ["UPSTAGE_API_KEY"] = upstage_api_key

# 사용자 정보 확인 및 메시지 표시
if people == 'YES':
    if not email:
        st.warning("당신의 게임 이메일을 입력해주세요.")
        st.stop()
elif people == 'NO':
    if not num:
        st.warning("당신의 사원번호를 입력해주세요.")
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
faiss_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(splits)

# EnsembleRetriever 구성
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],  # 사용할 검색 모델의 리스트
    weights=[0.7, 0.3],  # 각 검색 모델의 결과에 적용할 가중치
    search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
)

# prompt
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """
    당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
    당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
    검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 
    만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 
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

llm = ChatUpstage()

#대화 히스토리를 포함한 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the user's question: {context}"),
    ("human", "Here's our conversation history:\n{history}\n\nNow, please answer this question: {question}")
])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a helpful assistant for Netmarble's mobile game "The Second Country" (제2의 나라).
#     Your users are players and operators of this game. Always be polite, friendly, and supportive in your responses.
#     Use the following context to answer the user's question: {context}
#     If you don't know the answer or can't find it in the given context, honestly say "죄송합니다. 주어진 정보에서 해당 질문에 대한 답변을 찾을 수 없습니다." (I'm sorry, I couldn't find the answer to that question in the given information.)
#     Answer in Korean, but keep technical terms and names in their original form."""),
#     ("human", "Here's our conversation history:\n{history}\n\n제2의 나라에 대해 다음 질문에 답변해 주세요: {question}")
# ])


rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "history": lambda x: x["history"],
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain = (
#     {
#         "context": lambda x: retriever.get_relevant_documents(x["question"]),
#         "history": lambda x: x["history"],
#         "question": lambda x: x["question"]
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

class StreamChain:
    def __init__(self, chain):
        self.chain = chain
    
    def stream(self, query, history):
        response = self.chain.stream({"question": query, "history": history})
        complete_response = ""
        for token in response:
            yield token
            complete_response += token
        return complete_response

chain = StreamChain(rag_chain)

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant.")
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
        full_response = ""
        for response in chain.stream(prompt, history):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append(AIMessage(content=full_response))

st.empty()
