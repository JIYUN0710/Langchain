import streamlit as st
from langchain_openai import ChatOpenAI
import os

chat = ChatOpenAI(temperature=0)

# Streamlit UI 설정
st.set_page_config(page_title="ChatOpenAI Demo", page_icon=":robot:")
st.header("ChatOpenAI Demo")

# 사이드바에 OpenAI API 키 입력 필드 생성
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")


# 사용자 입력 처리
if openai_api_key :
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.info("please insert your key.")
    st.stop()


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
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in chat.stream(st.session_state.messages):
            full_response += (response.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 스크롤을 최하단으로 이동
st.empty()