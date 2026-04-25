# 1 历史消息
在 LangChain 中，带有历史消息的智能体通常通过 MessagesPlaceholder 和 ChatMessageHistory 实现。它将对话过程中的 HumanMessage（用户说的话）和 AIMessage（AI 回的消息）按顺序存储，并在下一次发起请求时，将这些历史记录重新塞进 Prompt 模具中发给大模型
```python
#  改进 Prompt：增加一个“占位符”来存放历史记录
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个全能助手。请参考历史对话来回答用户的问题。"),
    # variable_name 必须和后面调用时传的参数名一致
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "{input}")
])
```
在后面构建chain的时候，需要将**历史记录**和用户输入作为参数传递给chain
```python
# 2. 构建 Chain
# 此时 chain 的输入需要包含 'input' 和 'chat_history' 两个 key
chain = prompt | llm | parser
```

虽然 Prompt 里留了位置，但谁来负责往里面写东西呢？你有两个选择：
1. 前端维护（最灵活）
你的 Vue 3 前端维护一个数组 messages = []。每次用户发消息，前端把之前的历史记录全部传给后端。
2.  后端持久化（最专业）
使用 LangChain 内置的 BaseChatMessageHistory（如存储在 Redis 或 PostgreSQL 中）。
操作：你需要使用 **RunnableWithMessageHistory** 来包装你的 Chain。
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#这里定义了一个字典，用来存储每个session的历史记录
#然后定义了一个函数，用来根据session id获取历史记录
#最后这个函数在RunnableWithMessageHistory中被调用，用来获取历史记录
store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(model,get_session_history)

```

```python
config={"configurable":{"session_id":"chat1"}}
#question1
response=with_message_history.invoke(
    [HumanMessage(content="Hi , My name is Krish and I am a Chief AI Engineer")],
    config=config
)
response.content
#answer1
with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
```


这里切换了session id，所以历史记录会被清空，回答就不知道了
```python
## change the config-->session id

config1={"configurable":{"session_id":"chat2"}}
response=with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
response.content

```

```python
#question2
response=with_message_history.invoke(
    [HumanMessage(content="Hey My name is John")],
    config=config1
)
response.content
```
```python
#answer2
response=with_message_history.invoke(
    [HumanMessage(content="Whats my name")],
    config=config1
)
response.content
```

# 2 剪枝消息
**trim_messages** 是 LangChain 的一个工具组件，用于截断消息序列。它允许开发者基于 Token 数量或消息数量，结合特定策略（如保留最新消息、保留系统提示词等），动态管理发送给 LLM 的上下文窗口。

```python
from langchain_core.messages import SystemMessage,trim_messages
trimmer=trim_messages(
    max_tokens=2000,#最大token数量
    strategy="last",#保留最新消息
    token_counter=model,#用model自带的计算方法作为token计数器
    include_system=True,#每次都不能剪掉系统提示词
    allow_partial=False,#不允许剪掉一整条长消息 如果塞不进就整条丢弃
    start_on="human"#确保修剪后的第一条消息（除了系统消息外）必须是用户（Human）发的
)
messages = [
    SystemMessage(content="you're a good assistant"),
    
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)
```
通过 RunnablePassthrough 和 itemgetter 实现了一个全自动的、带内存清理的推理链。

看一下chain推理链是怎么组成的
1. itemgetter("messages")：在前面invoke参数传进来的messages中，准确的拿走messages键对应的值，也就是拿到了所有SystemMessage HumanMessage AIMessage
   ```python
   messages={
    "messages": [SystemMessage(...), HumanMessage(...), ...], # 这是一个列表
    "language": "English"                                      # 这是另一个字符串
   }
   ```
2. RunnablePassthrough.assign(...)：它的作用是“更新或添加”输入字典中的键值对。
   在这里，它执行了：new_messages = trimmer(old_messages)。然后再把new_messages赋值给messages键
3. 再重新组装成chain，包括messages prompt model
4. 重新调用的时候在messages键中添加messages字典+humanmessages就可以

```python
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain=(
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
    | prompt
    | model
    
)

response=chain.invoke(
    {
    "messages":messages + [HumanMessage(content="What ice cream do i like")],
    "language":"English"
    }
)
response.content
```
```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",#参数input_messages_key：指定输入字典中哪个键代表“当前最新的消息内容”，它会把从 get_session_history 拿到的历史列表，和你传入的最新消息，合并成一个完整的列表，然后以这个键名传递给后面的 chain
)
config={"configurable":{"session_id":"chat5"}}
```

最后调用打印 response结果
```python
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

response.content
```

# 3 使用retriever和vector store进行历史消息存储
代码详见history.ipynb[11],构建retriever有两种方法
## 1. 这里我们使用到了RunnableLambda对象(推荐)
```python

from langchain_core.runnables import RunnableLambda
# 这里优化了一下similarity_search_with_score方法 ，将其封装成一个RunnableLambda对象 支持 .batch()：自动开启多线程/异步处理。支持lcel语法：可以直接连接到 prompt 或 llm 后面。
retriever=RunnableLambda( db.similarity_search_with_score).bind(k=1)
retriever.batch(["气候","含氧量"])
```
## 2. 使用as_retriever方法
```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

results = retriever.batch(["气候", "含氧量"])
results
```

最后构建了一个 有状态的 RAG 链。它实现了一个并行的预处理步骤
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

messages = """
使用提供的文本回答我的问题
{question}

context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", messages)])
rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm
response=rag_chain.invoke("告诉我一些关于珠峰的气候的信息")
response.content
```
核心语法拆解：
###  字典定义 { "context": ..., "question": ... }
在 LangChain 中，如果你在管道的第一步写一个字典，它代表并行处理
1. context 这个键，会去调用你刚才定义的 retriever.invoke()。
2. question 这个键，会去接受invoke时候传进去的字符串，作为问题。

###  RunnablePassthrough()：
因为 invoke 的输入只有一个字符串。
如果没有它，你的 question 键就拿不到任何值。它就像一根透明的管子，把输入直接导向 prompt 里的 {question} 占位符。

运行rag_chain.invoke的时候  ->  
检索器会返回结果 ->  
prompt会被填充 -> 
然后llm再回答完整结果

