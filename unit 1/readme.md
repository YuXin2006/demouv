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

