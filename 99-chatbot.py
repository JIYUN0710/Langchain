# 필요한 라이브러리 임포트
from openai import OpenAI
import streamlit as st

# 사이드바에 OpenAI API 키 입력 필드 생성
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# 앱 제목 및 설명 설정
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# 세션 상태에 메시지 목록 초기화 (처음 실행 시)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# 저장된 대화 내용 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input():
    # API 키가 입력되지 않았을 경우 경고 메시지 표시
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=openai_api_key)
    
    # 사용자 메시지를 세션 상태에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # OpenAI API를 사용하여 응답 생성
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    
    # 생성된 응답을 세션 상태에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)