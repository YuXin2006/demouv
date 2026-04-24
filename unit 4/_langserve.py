from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
#from langchain_groq import ChatGroq
import os 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langserve import add_routes


load_dotenv()
os.environ['DEEPSEEK_API_KEY']=os.getenv('DEEPSEEK_API_KEY')
os.environ['DEEPSEEK_BASE_URL']=os.getenv('DEEPSEEK_BASE_URL')
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
silconflow_api_key=os.getenv("SILICONFLOW_API_KEY")

llm = ChatOpenAI(
    model_name='Pro/zai-org/GLM-4.7', 
    openai_api_key=os.environ["SILICONFLOW_API_KEY"],
    openai_api_base='https://api.siliconflow.cn/v1', # 关键：指向国内加速网关
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "把这句话翻译为{language}:"),
    ("user", "{input}")
])

parser = StrOutputParser()

chain=prompt|llm|parser

app=FastAPI(
    title="Langchain Translation",
    version="1.0",
    description="一个简单翻译文本的langchain接口",
)
add_routes(app, chain,path="/chain")
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
 