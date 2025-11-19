# 导入numpy模块
import numpy as np
# 将整数列表转换为NumPy数组
a = np.array([1,2,3])
# 查看数组对象
a


# 查看整数数组对象类型
a.dtype


# 将浮点数列表转换为NumPy数组
b = np.array([1.2, 2.3, 3.4])
# 查看浮点数数组对象类型
b.dtype


# 将两个整数列表转换为二维NumPy数组
c = np.array([[1,2,3], [4,5,6]])
c


# 生成2×3的全0数组
np.zeros((2, 3))


# 生成3×4的全1数组
np.ones((3, 4), dtype=np.int16)


# 生成2×3的随机数数组
np.empty([2, 3])


# arange方法用于创建给定范围内的数组
np.arange(10, 30, 5 )


# 生成3×2的符合(0,1)均匀分布的随机数数组
np.random.rand(3, 2)


# 生成0到2范围内长度为5的数组
np.random.randint(3, size=5)


# 生成一组符合标准正态分布的随机数数组
np.random.randn(3)


# 创建一个一维数组 
a = np.arange(10)**2
a


# 获取数组的第3个元素
a[2]


# 获取第2个到第4个数组元素
a[1:4]


# 一维数组翻转
a[::-1]


# 创建一个多维数组
b = np.random.random((3,3))
b


# 获取第2行第3列的数组元素
b[1,2]


# 获取第2列数据
b[:,1]


# 获取第3列前两行数据
b[:2, 2]


# 创建两个不同的数组
a = np.arange(4)
b = np.array([5,10,15,20])
# 两个数组做减法运算
b-a


# 计算数组的平方
b**2


# 计算数组的正弦值
np.sin(a)


# 数组的逻辑运算
b<20


# 数组求均值和方差
np.mean(b)


# 数组求方差
np.var(b)


# 创建两个不同的数组
A = np.array([[1,1],
              [0,1]])
B = np.array([[2,0],
              [3,4]])
# 矩阵元素乘积
A * B


# 矩阵点乘
A.dot(B)


# 矩阵求逆
np.linalg.inv(A)


# 矩阵求行列式
np.linalg.det(A)


# 创建一个3×4的数组
a = np.floor(10*np.random.random((3,4)))
a


# 查看数组维度
a.shape


# 数组展平
a.ravel()


# 将数组变换为2×6数组
a.reshape(2,6)


# 求数组的转置
a.T


a.T.shape


# -1维度表示NumPy会自动计算该维度
a.reshape(3,-1)


# 按行合并A数组和B数组
np.hstack((A,B))


# 按列合并A数组和B数组
np.vstack((A,B))


# 创建一个新数组
C = np.arange(16.0).reshape(4, 4)
C


# 按水平方向将数组C切分为两个数组
np.hsplit(C, 2)


# 按垂直方向将数组C切分为两个数组
np.vsplit(C, 2)


# 导入iris数据集和逻辑回归算法模块
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# 导入数据
X, y = load_iris(return_X_y=True)
# 拟合模型
clf = LogisticRegression(random_state=0,max_iter=1000).fit(X, y)
# 预测
clf.predict(X[:2, :])


# 概率预测
clf.predict_proba(X[:2, :])


# 模型准确率
clf.score(X, y)