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
st.set_page_config(page_title="제2의 나라 chat", page_icon=":video_game:")
st.header("제2의 나라 chat *^^*")

with st.sidebar:
    upstage_api_key = st.text_input("Upstage API Key", key="chatbot_api_key", type="password")
if upstage_api_key:
    os.environ["UPSTAGE_API_KEY"] = upstage_api_key
else:
    st.info("Upstage API Key를 입력해주세요.")
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

llm = ChatUpstage()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class StreamChain:
    def __init__(self, chain):
        self.chain = chain

    def stream(self, query):
        response = self.chain.stream(query)
        complete_response = ""
        for token in response:
            print(token, end="", flush=True)
            complete_response += token
        return complete_response


# 생성자에 chain 을 매개변수로 전달하여 chain 객체를 생성합니다.
chain = StreamChain(rag_chain)


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# 대화 히스토리 표시
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력 처리
prompt = st.chat_input("무엇을 도와드릴까요?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in chain.stream(st.session_state.messages):
            full_response += (response.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 스크롤을 최하단으로 이동
st.empty()