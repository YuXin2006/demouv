##这里使用streamlit做简单的页面 ，用户可以在页面上输入问题，模型会返回回答。
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()
os.environ['DEEPSEEK_API_KEY']=os.getenv('DEEPSEEK_API_KEY')
os.environ['DEEPSEEK_BASE_URL']=os.getenv('DEEPSEEK_BASE_URL')
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

st.title("简单聊天机器人")
##input=st.text_input("请输入您的问题") 改进 1
input=st.chat_input("请输入您的问题")
#这是控制面板 侧边栏
with st.sidebar:
    st.title("⚙️ 控制面板")
    model_choice = st.selectbox("选择大脑", ["DeepSeek", "Local-Gemma"])
    if st.button("🧹 清空对话"):
        st.session_state.messages = [AIMessage(content="对话已重置，请问有什么可以帮您？")]
        st.rerun()
    
    st.info("提示:DeepSeek 适合逻辑推理,Gemma 适合本地快速测试。")    

@st.cache_resource# 改进3:缓存模型，避免重复初始化
def init_llm():
    return ChatOpenAI(
        model_name='deepseek-chat',
        openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
        openai_api_base=os.getenv('DEEPSEEK_BASE_URL'),
        streaming=True,
    )
llm=init_llm()

prompt=ChatPromptTemplate.from_messages([
    ("system","你是一个专业的助手，帮助用户解决问题。"),
    ("user","问题是:{input}"),
])

output_parser=StrOutputParser()
chain=prompt|llm|output_parser
if input:
    st.write(chain.stream({"input":input}))# 改进 2 invoke->stream
else:
    st.write("您还没有输入问题，请输入您的问题")


    