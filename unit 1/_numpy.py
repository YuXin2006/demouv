import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[1,2,3],[7,8,9]])

print(arr1.shape)
print(arr2.reshape(3,2))

v1 = np.array([1, 2, 3])
v2 = np.array([1, 2, 2])

dot_product = np.dot(v1, v2)#dot表示计算向量的点积，返回标量结果

norm_v1 = np.linalg.norm(v1)#linalg.norm表示计算向量的模长norm
norm_v2 = np.linalg.norm(v2)

similarity = dot_product / (norm_v1 * norm_v2)#similarity表示计算向量的相似度
print(similarity)

arr1 = np.array([1, 2, 3])
arr2=np.array([4, 5, 6])
print(np.add(arr1,arr2))
print(np.sqrt(arr1))
print(np.exp(arr1))
print(np.log(arr1))

print(np.sum(arr1))
print(np.mean(arr1))
print(np.median(arr1))#median表示数组的元素求中位数，返回标量结果
print(np.std(arr1))#std表示数组的元素求标准差，返回标量结果




