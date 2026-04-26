# 1 tool
工具本质上是一个被“包装”过的 Python 函数。
一个工具的三个核心要素：(类似于函数的三要素)

名称 (Name)：比如 search_weather。

描述 (Description)：这是最重要的部分。模型通过阅读描述来决定是否使用该工具。用'''包裹起来的字符串'''。

参数模式 (Args Schema)：定义函数需要输入什么（比如是字符串还是数字）。
## 如何把函数变成一个工具

方式 A：使用 @tool 装饰器（最推荐）
```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘。""" # 这个文档字符串就是给 AI 看的说明书
    return a * b
```

方式 B：封装现有的 RAG 链
```python
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever, 
    "everest_knowledge_base",
    "搜索关于珠穆朗玛峰的气候、地理和历史信息。如果你遇到关于珠峰的问题，请使用此工具。"
)
```


## 工具调用的底层逻辑（Tool Calling）
1. 请求：你把 tools=[multiply] 传给模型。
2. 输出：模型不再返回普通的句子，而是返回一个 特定的 JSON 结构。
例如：{"name": "multiply", "arguments": {"a": 3, "b": 5}}
3. 执行：LangChain 框架看到这个 JSON，自动运行对应的 Python 函数。
4. 反馈：把函数的运行结果（15）再塞回给模型。

## 工具执行循环
工具执行循环并不是一次性完成的，它是一个 “观察 -> 思考 -> 行动 -> 观察” 的持续过程，直到问题被解决。

# 2 结构化输出
## typedict
```python
from typing import TypedDict
from langchain_openai import ChatOpenAI

# 1. 定义输出结构
class MovieInfo(TypedDict):
    name: str
    year: int
    director: str

llm = ChatOpenAI(model="Pro/zai-org/GLM-4.7")

# 2. 绑定结构化输出（1.2.x 推荐写法）
structured_llm = llm.with_structured_output(MovieInfo)

# 结果将直接是一个 Python 字典：{'name': '盗梦空间', 'year': 2010, 'director': '诺兰'}
result = structured_llm.invoke("介绍一下电影盗梦空间")
```

## dataclass
```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

@dataclass
class Person:
    name: str
    age: int

llm = ChatOpenAI(model="Pro/zai-org/GLM-4.7")
structured_llm = llm.with_structured_output(Person)

# 结果是一个 Person 对象：Person(name='YuXin', age=25)
result = structured_llm.invoke("我叫YuXin，今年25岁")
print(result.name) # 可以直接点出属性
```

## pydantic
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. 定义极其严格的结构
class ITWorkOrder(BaseModel):
    title: str = Field(description="工单标题")
    priority: int = Field(description="优先级 1-5，5 最高", ge=1, le=5)
    tags: list[str] = Field(default=[], description="技术标签，如 Vue, Django")

llm = ChatOpenAI(model="Pro/zai-org/GLM-4.7")
structured_llm = llm.with_structured_output(ITWorkOrder)

# 即使 LLM 吐出的 JSON 稍微有点瑕疵，Pydantic 也会尝试修复或报错拦截
result = structured_llm.invoke("帮我创建一个高优先级的 Vue 开发任务")
```

# 3 中间件
我们通常不叫它“中间件”，而叫 Callbacks (回调) 或 Custom Runnables。它们负责在 LLM 生成内容的瞬间进行干预。以下有列出了一些常用的中间件类型：
## Callbacks (回调)
例如这是一个敏感词过滤中间件：
```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

class ContentFilterMiddleware(BaseCallbackHandler):
    """自定义中间件：发现 AI 输出敏感词时进行警告"""
    def on_llm_end(self, response, **kwargs):
        text = response.generations[0][0].text
        if "禁止词汇" in text:
            print("⚠️ 警告：检测到模型输出不当内容！")

llm = ChatOpenAI(model="Pro/zai-org/GLM-4.7")
# 在调用时挂载“中间件”
llm.invoke("你好", config={"callbacks": [ContentFilterMiddleware()]})
```

## custom runnable（lcel拦截器）：数据流中间件
例如这是一个添加上下文中间件：
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

def add_user_role_middleware(input_data):
    """中间件：在所有问题前自动给 AI 加上用户职位的上下文"""
    input_data["question"] = f"我是前端工程师，请回答：{input_data['question']}"
    return input_data

prompt = ChatPromptTemplate.from_template("{question}")
chain = RunnableLambda(add_user_role_middleware) | prompt | llm

# 即使只输入“怎么用 Vue”，中间件也会将其改为“我是前端工程师...”
chain.invoke({"question": "怎么用 Vue？"})
```

## 内置中间件（Built-in middleware）：剪枝器
LangChain 官方内置了强大的消息修剪器。它的作用像是一个“自动滑窗中间件”，确保发送给模型的 Token 永远不会超过限制。
```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# 定义一个“修剪中间件”逻辑
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",          # 保留最后的对话
    token_counter=ChatOpenAI(model="Pro/zai-org/GLM-4.7"),
    include_system=True,      # 永远保留系统提示词
    start_on="human",         # 确保从用户说话开始，保持对话结构
)

# 在 Chain 中像中间件一样使用
chain = trimmer | llm
```
## 总结中间件(summarization middleware)：压缩器 (Summarization)

总结中间件比简单的裁剪更高级。它不是直接丢弃旧消息，而是将旧的对话压缩成一段摘要
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage

def summarization_middleware(input_data):
    history = input_data["history"]
    # 逻辑判断：如果历史消息超过 5 条，就触发总结
    if len(history) > 5:
        # 这里会调用一次轻量级 LLM 进行总结
        summary = summarize_llm.invoke(f"总结以下对话：{history}")
        # 返回处理后的“中间件结果”
        return {"history": [SystemMessage(content=f"之前的背景：{summary}")], "input": input_data["input"]}
    return input_data

# 组装链：中间件放在最前面
chain = RunnablePassthrough.assign(processed=summarization_middleware) | prompt | llm
```

## Human-in-the-loop  中间件：人机交互
在大模型完全自动化的 Agent 循环中，有些操作是不可逆或高风险的（例如：在 OA 系统中给全公司发工资、在数据库执行删除）。

我们通常在 on_tool_start 阶段通过阻塞式输入（在 Web 端则是通过状态机）来实现。



