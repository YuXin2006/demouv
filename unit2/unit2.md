整个rag的组件分成两个模块

前半段:数据摄入->数据切片(data转化为text chunks)->嵌入(text转化为vector)->存储到向量数据库->检索

后半段:question  ->  retrive chain(api=>db)  ->  prompt  ->  llm  ->  answer ->  output

# 1 加载
数据摄入涉及三个维度:
1. 静态摄入 例如 pdf word txt markdown 关键技术:文件解析
2. 动态摄入 例如 网页数据 新闻 实时天气 关键技术:爬虫和提取
3. 系统摄入 例如 数据库 ERP日志系统 关键技术：SQl接口和API适配

# 2 切片
## 2.1 字符文本
通常使用的是*RecursiveCharacterTextSplitter*
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "这里是第一段。这里有很多细节...\n\n这里是第二段。Agent开发很有趣。"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # 每个块的目标字符数
    chunk_overlap=20,    # 块与块之间的重叠，防止语义在切割点断掉
    separators=["\n\n", "\n", "。", " ", ""] # 切割优先级
)

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
```
## 2.2 html标题
通常使用的是*HTMLHeaderTextSplitter*
```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <h1>AI 核心技术</h1>
    <p>这是关于AI的总体介绍。</p>
    <h2>机器学习</h2>
    <p>机器学习是AI的一个分支，包括监督学习和无监督学习。</p>
    <h2>深度学习</h2>
    <p>深度学习使用神经网络处理复杂数据。</p>
</body>
</html>
"""

# 定义你想要追踪的标题层级
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)

for chunk in html_header_splits:
    # chunk.page_content 是正文，chunk.metadata 会包含所属标题
    print(f"内容: {chunk.page_content}")
    print(f"元数据: {chunk.metadata}\n")
```

## 2.3 json文本
核心逻辑： 深度优先遍历 JSON，在保持结构完整的前提下，将大的 JSON 拆分成小的子 JSON。通常使用*RecursiveJsonSplitter*
```python
from langchain_text_splitters import RecursiveJsonSplitter

# 假设这是一个复杂的商品或用户信息 JSON
data = {
    "store": {
        "book": [
            {"category": "reference", "author": "Nigel Rees", "title": "Sayings of the Century", "price": 8.95},
            {"category": "fiction", "author": "Evelyn Waugh", "title": "Sword of Honour", "price": 12.99}
        ],
        "bicycle": {"color": "red", "price": 19.95}
    }
}

# 设置每个 JSON 块的最大字符数
json_splitter = RecursiveJsonSplitter(max_chunk_size=150)

# 获取切分后的 JSON 对象列表
json_chunks = json_splitter.split_json(json_data=data)

# 或者直接获取切分后的字符串列表
json_string_chunks = json_splitter.split_text(json_data=data)

for i, chunk in enumerate(json_string_chunks):
    print(f"JSON Chunk {i+1}:\n{chunk}\n")
```    
# 3 嵌入(embedding)
## 3.1 OPENAI(付费)
首先安装 langchain_openai和python_dotenv。这是目前最流行的商业方案。你需要一个 OPENAI_API_KEY，优点是速度快、模型能力极强，不需要占用本地电脑资源。
先把api key写在.env的文件中 然后在词嵌入的文件中读取密钥
```python
#.env
OPENAI_API_KEY=""
```
再读取环境变量
```python
import os
from dotenv import load_dotenv
load_dotenv()#读取所有的环境变量
os.environ["OPENAI_API_KEY"]=os.getenv["OPENAI_API_KEY"]
```
导入openai自带的embedding工具
```python
from langchain_openai import OpenAIEmbeddings

embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
embeddings
```
再把这个embedding模型应用到text chunks上
```python
text=""
query_result=embeddings.embed_query(text)#这里的query_result是一个vector,包含3072个元素的列表
```
也可以自定义维度
```python
embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)
text=""
query_result_1024=embeddings_1024.embed_query(text)#这里的query_result_1024是一个vector,包含1024个元素的列表        
```
## 3.2 ollama(开源)
先下载ollama文件 然后在终端中安装ollama模型
```bash
ollama run gemma:2b
```
```python
from langchain_community.embeddings import OllamaEmbeddings
embeddings=(
    OllamaEmbeddings(model="gemma:2b")
)
r1=embeddings.embed_documents(
    [
        "珠穆朗玛峰是世界第一高峰",
        "珠穆朗玛峰是世界最高的山",
    ]
)
r2=embeddings.embed_documents(
    ["珠穆朗玛峰很高"]
)
```
这里我们就得到了r1 包含两条语句嵌入后的vector，r2 包含了一条语句嵌入后的vector


```python
import numpy as np
dot_product=np.dot(r1[1],r2[0])
norm_v1 = np.linalg.norm(r1[1])
norm_v2 = np.linalg.norm(r2[0])
cosine_similarity = dot_product / (norm_v1 * norm_v2)
cosine_similarity
round(cosine_similarity.item(), 4)
```
最后用numpy的向量运算计算余弦相似度,判断两条语句的相似度，最后格式化输出，保留4位小数
## 3.3 hugging face（开源）

# 4 存储
可以使用三种向量存储工具 :*Faiss*,*ChromaDB*,*
# 4.1 Faiss
faiss有三个关键词:向量存储 索引(核心) 距离度量

1. 数据量 < 1万条 IndexFlatL2 简单、精准、无需训练。

2. 内存有限，数据量大 IndexIVFFlat 速度快，通过聚类减少计算量。

3. 追求极致的搜索性能 IndexHNSW 响应时间极短，是大多数 RAG 应用的首选。

4. 数据量巨大（千万级以上） IndexIVFPQ 引入了“乘积量化”（PQ），能把向量压缩，节省大量内存。
这里是使用的示例代码
```python
import faiss

# 假设维度为 d
d = 1536 

# 1. Flat 索引
index = faiss.IndexFlatL2(d)

# 2. IVF 索引 (需要训练)
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
# index.train(data)

# 3. HNSW 索引
index = faiss.IndexHNSWFlat(d, 32)

# 4. IVFPQ 索引 (压缩向量)
m = 8 # 每个向量被压缩成 8 个字节
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
```
```python
from langchain_community.vectorstores import FAISS
db=FAISS.from_documents(docs, embeddings)
db
```
这里使用chromaDB作为数据库
```python
from langchain_community_vectorstores import Chroma
vector_db = Chroma.from_documents(
    documents=chunks,# 文档列表
    embedding=embeddings_model,# 嵌入模型
    persist_directory="./my_agent_db" # 数据持久化到本地路径
)
```

# 5 检索
1. 可以直接进行向量检索(最底层的检索方式)
```python 
# 2. 模拟 Agent 的检索行为
query = "什么是 AI Agent？"
# 在数据库中寻找最相关的 2 条记录
search_results = vector_db.similarity_search(query, k=2)

print("\n--- 检索结果 ---")
for i, res in enumerate(search_results):
    print(f"结果 {i+1}: {res.page_content}")

```    
2. 也可以使用检索器，把向量存储数据库转化为检索器（封装后的检索方式）
```python
query="什么是AI Agent？"
retriever=vector_db.as_retriever()
search_results=retriever.invoke(query)#这样可以直接嵌入rag系统中
print(search_results)
```





