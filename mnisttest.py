import struct
import numpy as np
import time
from sklearn import svm
#from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier



def loadarr(path, isImage):
	with open(path, 'rb') as f:
		f.read(16 if isImage else 8)
		
		if (isImage):
			return np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)
		else:
			return np.fromfile(f, dtype=np.uint8)

imagearr = loadarr('mnistdata/train-images.idx3-ubyte', True)
labelarr = loadarr('mnistdata/train-labels.idx1-ubyte', False)

imagearrtest = loadarr('mnistdata/t10k-images.idx3-ubyte', True)
labelarrtest = loadarr('mnistdata/t10k-labels.idx1-ubyte', False)

"""
# 显示图像
imageReshape = imagearr.reshape(-1,28,28)
"""

"""
# 显示多幅图像
plt.figure(figsize=(10,10))
plt.suptitle("imagessss")

for i in range(20):
	plt.subplot(4,5,i+1)
	plt.title('image')
	plt.imshow(imageReshape[i], cmap='gray')
	plt.axis("off")

plt.show()
"""

"""
# 显示单幅图像
plt.figure("Image")
plt.imshow(imageReshape[1] ,cmap='gray')
plt.axis("on")
plt.title("image")
plt.show()
"""


print(imagearr)
print(imagearr.shape)
print(labelarr)
print(labelarr.shape)

scaler = StandardScaler().fit(imagearr)
#imagearr = scaler.transform(imagearr)
#imagearrtest = scaler.transform(imagearrtest)


#help(svm)

#clf = svm.LinearSVC(max_iter=10000)
if (os.path.isfile("mymodel.m")):
	print('从硬盘加载模型')
	clf = joblib.load("mymodel.m")
else:
	print('重新训练模型')
	
	
	clf = svm.SVC()
	#clf = svm.SVC(kernel='linear')
	#clf = svm.LinearSVC(max_iter=1000)
	
	
	#help(scaler)
	
	
	t1 = time.time()
	clf.fit(imagearr[:60000:20], labelarr[:60000:20])
	joblib.dump(clf, "mymodel.m")
	t2 = time.time()
	print(t2 - t1, '秒')
#help(clf)
print(clf)
p = clf.predict(imagearrtest[1000:1050])
print(p)



#score = clf.score(imagearrtest, labelarrtest)
#print(score)


# 使用不同分类器进行训练，保存模型，并打印训练时间、成功率
clfs = (\
	#{'name':'svc-rbf','clf':svm.SVC()},\
	#{'name':'svc-linear-kernel', 'clf':svm.SVC(kernel='linear'), 'scale': True},\
	# 训练时间很久
	#{'name':'linearsvc', 'clf':svm.LinearSVC(max_iter=60000), 'scale': True},\
	# 测试时间久
	#{'name':'KNeighborsClassifier', 'clf':KNeighborsClassifier(), 'scale': False},\
	# 测试时间久
	#{'name':'KNeighborsClassifierScale', 'clf':KNeighborsClassifier(), 'scale': True},\
	# 使用未scale的数据训练时间久
	#{'name':'LogisticRegression', 'clf':LogisticRegression(max_iter=10000), 'scale': False},\
	#{'name':'LogisticRegressionScale', 'clf':LogisticRegression(max_iter=10000), 'scale': True},\
	#{'name':'RandomForestClassifier', 'clf':RandomForestClassifier(), 'scale': False},\
	#{'name':'RandomForestClassifier', 'clf':RandomForestClassifier(), 'scale': True},\
	#{'name':'DecisionTreeClassifier', 'clf':tree.DecisionTreeClassifier(), 'scale': False},\
	#{'name':'DecisionTreeClassifierScaler', 'clf':tree.DecisionTreeClassifier(), 'scale': True},\
	{'name':'GradientBoostingClassifier', 'clf':GradientBoostingClassifier(), 'scale': False},\
	{'name':'GradientBoostingClassifierScaler', 'clf':GradientBoostingClassifier(), 'scale': True},\
	
)
for c in clfs:
	print('-' * 20)
	print('分类器名:', c['name'])
	csub = c['clf']
	
	trainarr = imagearr
	
	if (c.get('scale')):
		print('scaler处理训练数据')
		trainarr = scaler.transform(imagearr)
	
	
	t1 = time.time()
	csub.fit(trainarr[:60000:10], labelarr[:60000:10])
	t2 = time.time()
	print('训练了', t2 - t1, '秒')
	joblib.dump(csub, c['name'] + ".m")
	
	testarr = imagearrtest
	if (c.get('scale')):
		print('scaler处理测试数据')
		testarr = scaler.transform(imagearrtest)
	
	t1 = time.time()
	score = csub.score(testarr, labelarrtest)
	t2 = time.time()
	print('测试了', t2 - t1, '秒')
	print('准确率:', score)
	print('-' * 20)


