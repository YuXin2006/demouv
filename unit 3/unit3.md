在开始使用langchain之前，我们需要先配置好环境变量。
```python
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
```
然后调用openai的模型
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm.invoke("什么是Langchain")
```
# 1 chain组件的组装
在langchain中 最终的输出=提示词模板+模型llm+输出解释器
## 1.1 提示词模板prompt template
在写程序时，我们不能硬编码对话。模板允许我们定义一个变量，动态地填入内容。核心作用就是将用户的输入转化为模型能理解的格式化指令。
```python
from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","你是一个人工智能专家"),
        ("user","{input}")
    ]
)
prompt_value=prompt.invoke({"input":"什么是Langchain"})
```
## 1.2 模型llm
在chain中，我们通常会将提示词模板和模型llm组合起来，形成一个完整的chain。
```python
chain=prompt|llm
response=chain.invoke({"input":"什么是Langchain"})
type(response)#class 'langchain_core.messages.ai.AIMessage'
```
## 1.3 输出解释器
```python
from langchain_core.output_parsers import StrOutputParser
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
response=chain.invoke({"input":"什么是Langchain"})
response
```

这样子 直接调用chain的逻辑就是:**输入->提示词模板->模型llm->输出解释器->最终输出**

# 2 检索链和文档链
在 Agent 开发中，检索链 (Retrieval Chain) 和 文档链 (Document Chain) 是 RAG（检索增强生成）流程中的两个关键组件。
## 2.1 检索链
input->retrieve_chains->vectordb->output 
检索链类似一个接口 直接根据输入从向量数据库中检索 并返回输出
检索链由retriver和document_chain组成
```python
retriver=storedb.as_retriever()
from langchain_classic.chains import create_retrieval_chain
retrieve_chain=create_retrieval_chain(retriver, document_chain)
retrieve_chain.invoke({"input":"什么是Langchain"})
```

## 2.2 文档链
文档链的核心任务是：处理一组已经找好的文档（List of Documents），并将它们塞进 Prompt 里让 LLM 总结或回答。


最常用的方式是 create_stuff_documents_chain。注意这里的create_stuff_documents_chain是langchain_classic中的，而不是langchain.chains.combine_documents中的。
```python
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template( """
根据以下上下文回答问题：
<context>
{context}
</context>
问题: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
```

# 3返回响应
```python
response=retrieve_chain.invoke({"input":"什么是Langchain"})
response["answer"]
```