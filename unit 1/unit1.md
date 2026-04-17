# 1.numpy基础
## 1.1 维度与形状 
在 Agent 处理文本搜索时，会将一句话转成一串数字（向量）。理解这串数字的结构至关重要。
shape属性和reshape方法表示数组的形状和维度 arange方法类似于range函数 生成等差数列的数组
``` python
arr1.shape()#shape表示数组的形状，(3,)表示长度为3一维数组，(2,3)表示2行3列的二维数组
arr1.reshape(3,2)#reshape表示将数组的形状改变为(3,2)，即3行2列的二维数组

np.arange(10)#arange表示生成从0到9的整数数组
np.arange(1,10,2)#arange表示生成从1到9的步长为2的整数数组,类似range函数
```

## 1.2 向量相似度
Agent 为什么能根据你的问题找到相关的文档？因为它计算了余弦相似度。这涉及到 NumPy 的点积 (Dot Product)。向量相似度涉及了numpy的点积和模长norm概念
``` python
v1 = np.array([1, 2, 3])
v2 = np.array([1, 2, 2])

dot_product = np.dot(v1, v2)#dot表示计算向量的点积，返回标量结果

norm_v1 = np.linalg.norm(v1)#linalg.norm表示计算向量的模长norm
norm_v2 = np.linalg.norm(v2)

similarity = dot_product / (norm_v1 * norm_v2)#similarity表示计算向量的相似度
```
## 1.3 基本数学操作
在 NumPy 中，你可以对数组进行基本的数学操作
``` python
arr1 = np.array([1, 2, 3])
arr2=np.array([4, 5, 6])
np.add(arr1,arr2)
np.sqrt(arr1)
np.exp(arr1)
np.log(arr1)
```
## 1.4 基本统计学操作
在 NumPy 中，你可以对数组进行基本的统计学操作，如求和、求平均值、求中位数、求标准差等。
``` python
np.sum(arr1)
np.mean(arr1)
np.median(arr1)#median表示数组的元素求中位数，返回标量结果
np.std(arr1)#std表示数组的元素求标准差，返回标量结果
```
## 1.5 广播机制
Agent 有时需要批量处理数据（比如给 100 个搜索结果统一打分）。广播机制让你不需要写 for 循环，速度极快。
``` python
scores = np.array([0.5, 0.8, 0.3])
bonus = 0.1

# 所有的分数同时加 0.1，NumPy 会自动“广播”这个操作
final_scores = scores + bonus 

print(final_scores) # [0.6, 0.9, 0.4]
```
## 1.6 索引与切片(布尔索引)
```python 
results = np.array([0.95, 0.42, 0.88, 0.30])

results = np.array([0.95, 0.42, 0.88, 0.30])

# 一行代码筛选出高价值结果
top_results = results[results > 0.8]

print(f"高相关度结果: {top_results}")
```

# 2.pandas基础
在 Agent 开发中，我们很少处理单一的数字，更多的是处理结构化数据：比如从数据库导出的用户信息、CSV 格式的知识库、或者是 Agent 运行过程中的轨迹日志（Tracing Logs）。
## 2.1 series
Pandas Series 不仅仅是一个一维数组，它通常被用作 Agent 的记忆单元（Memory Slot）、状态记录器（State Tracker）或环境观察向量（Observation Vector）。
这样子初始化一个Series
```python
data=[1,2,3,4,5]
series=pd.Series(data)
print(series)
```

也可以自定义索引
``` python
index=['a','b','c','d','e']
series=pd.Series(data,index=index)
print(series)
```
## 2.2 DataFrame
DataFrame 是一个二维表结构

可以这样初始化一个DataFrame(从字典初始化)
```python
data={
    'name':['张三','李四','王五','赵六','王二'],
    'age':[18,20,22,24,26],
    'gender':['男','男','女','女','男'],
}
df=pd.DataFrame(data)
print(df)
```
也可以从列表初始化
```python
data=[
    ['张三',18,'男'],
    ['李四',20,'男'],
    ['王五',22,'男'],
    ['赵六',24,'女'],
    ['王二',26,'男'],
]
df=pd.DataFrame(data,columns=['name','age','gender'])
print(df)
```
以上的DataFrame都是这个
| name   |   age | gender   |
|:-------|------:|:---------|
| 张三   |    18 | 男       |
| 李四   |    20 | 男       |
| 王五   |    22 | 男       |
| 赵六   |    24 | 女       |
| 王二   |    26 | 男       |

### 2.2.1 DataFrame的索引与切片
DataFrame的索引是行索引，默认是整数索引，也可以自定义索引。注意：iloc表示的位置不包含索引和标签
```python
print(df['age'])
print(df.loc['a'])
print(df.iloc[1,1])
```
### 2.2.2 DataFrame的增删改查
```python
#增列
df['selery']=df['age']*1000
print(df)   
#增行
df.loc['d']=['王二',26,'男',26000]
print(df)
#插入行
df.insert(2,'height',[1.7,1.8,1.9,1.8,1.7])
print(df)

#删列
df.drop('height',axis=1,inplace=True)
print(df)
#删行
df.drop('5',axis=0,inplace=True)
print(df)

#修改值
df.loc['a','age']=20
print(df)
#修改多行值
df.loc[['a','c'],'age']=[20,22]
print(df)
#修改列名
df.rename(columns={'age':'new_age'},inplace=True)
print(df)
#修改行索引
df.rename(index={'a':'A'},inplace=True)
print(df)
#按条件修改
df.loc[df['age']>20,'age']=20
print(df)

#查行
print(df.loc['A'])
print(df.iloc[1])
#查列
print(df['new_age'])
#查多列
print(df[['new_age','gender']])
#条件查询------
print(df[df['new_age']>20])
print(df[df['gender']=='男'])
```
## 2.3 pandas的数据操作
### 2.3.1 缺失值处理
缺失值是指在数据中缺少的值，通常用 NaN 表示。在 pandas 中，你可以使用 isnull() 方法检查缺失值，使用 dropna() 方法删除缺失值，使用 fillna() 方法填充缺失值。
```python
df.isnull()
df.dropna()
df.fillna(0)
```
也可以用isnull().sum()方法求缺失值的数量
```python
df.isnull().sum()
```
### 2.3.2 聚合统计
```python
df.mean()
df.median()
df.describe()#求统计信息
df.groupby('gender').mean()#按性别分组求平均值
```

### 2.3.3 合并与连接操作
```python
data.merge(df1,df2,on='name',how='inner')#内合并(交集)  
data.merge(df1,df2,on='name',how='outer')#外合并(并集)
data.merge(df1,df2,on='name',how='left')#左合并
data.merge(df1,df2,on='name',how='right')#右合并
data.merge(df1,df2,on='name',how='cross')#交叉合并
```
# 3. 使用pandas进行数据读取
## 3.1 读取CSV文件
先创建一个generate_data.csv文件用于生成一千条模拟数据的CSV文件
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df)
```
以下是read_csv()方法的参数
```python
pd.read_csv(
    "data.csv",
    sep=",",        # 分隔符
    header=0,      # 第 0 行是列名
    index_col=0,   # 第 0 列作为行索引
    encoding="utf-8"
)   
```
还可以使用usecols参数指定要读取的列,优化性能
```python
df = pd.read_csv("data.csv",usecols=["name", "score"])
print(df)
```
### 对于csv大文件读取
nrows参数可以指定只读取前n行数据，避免内存溢出
```python
df = pd.read_csv("big.csv", nrows=1000)#读取前1000行
```
对于超大文件，chunksize参数可以指定每次读取的行数，避免内存溢出
```python
chunk_iter = pd.read_csv("big.csv", chunksize=10000)
for chunk in chunk_iter:
    print(chunk.shape)
```

## 3.2 读取Excel文件
```python
df = pd.read_excel("large_sales_data.xlsx")
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")#注意指定sheet_name参数，默认读取第一个sheet
```
和csv一样也可以使用nrows或参数指定只读取前n行数据，避免内存溢出
```python
df = pd.read_excel("big.xlsx", nrows=1000)#读取前1000行
```
## 3.3 读取JSON文件
基本用法:
```python
df = pd.read_json("large_sales_data.json")
print(df)
```
### 3.3.1 json常见结构的读取方法
对于列表型json
```python
[
  {"A": 1, "B": 2},
  {"A": 3, "B": 4}
]
pd.read_json("data.json")#默认读取列表型json
print(df)
```
对于嵌套型json
```python
[
  {
    "id": 1,
    "info": {"name": "张三", "age": 18}
  }
]
pd.json_normalize(data)
```
对于键值对型json
```python
{
  "张三": {"age": 18, "score": 90},
  "李四": {"age": 19, "score": 85}
}
df = pd.read_json("data.json")
df = df.T   # 转置
print(df)
```
### 3.3.2 orient 
一句话先记住 orient告诉 Pandas：JSON 的外层结构，是按「行」组织，还是按「列」组织。
orient参数可以指定json文件的取向，默认是"columns"，也可以指定为"index"、"records"、"split"、"table"等

records: 标准json结构,如web api
```python
[
  {"name": "张三", "age": 18},
  {"name": "李四", "age": 19}
]
pd.read_json("data.json",orient="records")
```

index: 每个json对象的key作为行索引
```python
{
  "row1": {"name": "张三", "age": 18},
  "row2": {"name": "李四", "age": 19}
}
pd.read_json("data.json",orient="index")
```
columns: json对象的key作为列索引

table: 以表格形式组织数据，类似数据库表的结构
```python
{
  "schema": {
    "fields": [
      {"name": "name", "type": "string"},
      {"name": "age", "type": "integer"}
    ],
    "primaryKey": ["name"]
  },
  "data": [
    {"name": "张三", "age": 18},
    {"name": "李四", "age": 19}
  ]
}
pd.read_json("data.json",orient="table")
```


## 3.4 读取SQL数据库
这是核心函数，sql参数是SQL语句，con参数是数据库连接对象。
```python
pd.read_sql(sql, con)
```
### 读取mysql/postgresql数据库
#### 1安装依赖
```python
pip install sqlalchemy pymysql
```
#### 2创建连接
```python
from sqlalchemy import create_engine

engine = create_engine(
    "mysql+pymysql://user:password@localhost:3306/testdb"
)
```
#### 3读取数据
```python
df = pd.read_sql(
    "SELECT * FROM orders LIMIT 1000",
    engine
)
```

### read_sql()常用参数
```python
pd.read_sql(
    sql="SELECT * FROM table",
    con=engine,
    index_col="id",        # 指定索引
    coerce_float=True,     # 自动转 float
    parse_dates=["created_at"]
)
```
### 对于大量数据读取
chunksize参数可以指定每次读取的行数，避免内存溢出
```python
for chunk in pd.read_sql(
    "SELECT * FROM big_table",
    engine,
    chunksize=10000
):
    print(chunk.shape)
```    
只读取指定列数据
```python
pd.read_sql(
    "SELECT id, name FROM users",
    engine
)
```
## 3.5 读取HTML文件
```python
df = pd.read_html("data.html")
```
# 4. logging配置
## 4.1基础配置
```python
import logging

# 基础配置
logging.basicConfig(
    level=logging.DEBUG, # 设置级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w',
    filename='app.log'
)

logging.debug("this is a debug msg")
```
logging的级别从debug向上递增 级别低的可以执行级别高的操作

## 4.2 在agent开发中的基础日志配置
```python 
import logging

# 基础配置
logging.basicConfig(
    level=logging.INFO, # 设置级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("AgentLogger")

logger.info("Agent 启动完成...")
logger.error("API 调用超时！")
```
## 4.2 在agent开发中的进阶日志配置：多处理器
Agent 通常需要两套日志：一套在终端实时看进度，一套存进文件做离线评估。
```python
import logging

# 1. 创建 Logger 对象
logger = logging.getLogger("MasterAgent")
logger.setLevel(logging.DEBUG) # 总开关设为最低

# 2. 创建“终端”处理器（显示 INFO 以上）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 3. 创建“文件”处理器（记录 DEBUG 详细信息，用于复盘）
file_handler = logging.FileHandler('agent_debug.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 4. 设置不同的格式
formatter = logging.Formatter('%(name)s [%(levelname)s] -> %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 5. 添加到 Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug("这是给开发者看的详细 Prompt 调试信息")
logger.info("这是给用户看的任务执行进度")
```

# 5.pydantic
如果说 LLM 是一个“满嘴跑火车”的艺术家，Pydantic 就是那个手持表格、严格执法的“质检员”。在 Agent 架构中，它负责将模型输出的模糊文本，强制转化为代码可以直接运行的结构化数据。

pydantic就像带检验功能的detaclass，有三大核心功能:*自动类型转换*  *自动检验*  *拒绝多余字段* 

可以帮助检验数据是否正确或者把脏数据转化为正确格式的数据
## 5.1 最小示例
```python 
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    age: int = 18  # 默认值
u = User(id=1, name="张三")
print(u)
```    
## 5.2 optional
可以使用optional 指定字段默认值
``` python
from typing import Optional
class User(BaseModel):
    id:int
    name:str
    age:int
    salary:Optional[float]=None#默认为none
    is_active:Optional[bool]=True#默认为true
u=User(id=1,name='李四',age=18)
print(u)    
```