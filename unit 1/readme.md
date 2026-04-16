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

