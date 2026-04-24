# LCEL语言
首先搞清楚prompts和message的关系
1. prompts是一个函数或者模板 它用于接受用户输入的变量 并按照预设的结构生成一组message（类似于vue的html标签）
2. message是一个对象 它包含了用户输入的内容（或者模型的输出）以及它的类型（例如HumanMessage、AIMessage等） （类似于vue的template插槽）

再看一下chain链的lcel写法:
```python
chain= prompt | llm | parser
```
这里的chain链是一个流水线 它将prompts、llm、parser这三个组件按顺序连接起来
1. prompt组件负责将用户输入的变量转换为一组message
2. llm组件负责根据message生成模型的输出
3. parser组件负责将模型的输出转换为可解析的格式
   
最后调用的时候 只需要调用chain链的invoke方法 并传入用户输入的变量即可 注意：这里的参数是字典格式的在prompt组件中会被用于填充模板中的占位符
### 场景一：单变量
```python
chain.invoke({"input": "你好"})
```
### 场景二：多变量
```python
chain.invoke({
    "input": "你好", 
    "name": "YuXin"
})
```
### 场景三 带有“记忆槽位” (MessagesPlaceholder)
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。"),
    MessagesPlaceholder(variable_name="history"), # 注意这个变量名
    ("user", "{input}")
])
```
此时你的 invoke 就需要包含当前问题和历史列表：
```python
chain.invoke({
    "input": "那它怎么用呢？",
    "history": [HumanMessage(content="什么是LangChain?"), AIMessage(content="它是...")]
})
```
# langserve
## 1 专业定义
LangServe 是一个旨在帮助开发者将 LangChain 的 Runnable（链、模型等）部署为 REST API 的库。它基于 FastAPI 构建，并利用 pydantic 进行数据验证。它能自动为你的 Chain 生成标准的 API 接口（如 /invoke, /stream），并提供一个内置的交互式调试界面（Playground）。

先安装fastapi uvicorn langserve
```bash
pip install fastapi 
pip install uvicorn
pip install langserve
```
Uvicorn 是一个超快速的 ASGI (Asynchronous Server Gateway Interface) 服务器实现的web工作服务器，基于 uvloop 和 httptools 构建。它旨在为 Python 提供高性能的异步 Web 服务支持。它主要处理单个进程内的异步请求循环。（厨子）

Gunicorn (Green Unicorn) 是一个标准的 WSGI (Web Server Gateway Interface) HTTP 服务器，采用 Pre-fork 工人模型（Pre-fork Worker Model）。它主要充当“进程管理器”，负责启动、管理和监控多个工作进程（Workers），以实现并发处理和服务器稳定性。（大堂经理）

接下来写了一个翻译文本的langchain接口 代码如下：unit4/langserve.py
运行文件
```bash
python langserve.py
```
直接启动报错的话要下载sse_starlette依赖
```bash
pip install sse-starlette
```
然后跳转到http://localhost:8000/docs页面即可

后面就可以用postman测试接口 然后复制并补充请求体就可以得到格式化输出（json）
