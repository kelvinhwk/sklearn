import struct
import numpy as np
import time
from sklearn import svm
import joblib
from sklearn.preprocessing import StandardScaler
import os
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image
import centerimg

# 加载训练数据
def loadarr(path, isImage):
	with open(path, 'rb') as f:
		f.read(16 if isImage else 8)
		
		if (isImage):
			return np.fromfile(f, dtype=np.uint8).reshape(-1, 28*28)
		else:
			return np.fromfile(f, dtype=np.uint8)

# 训练数据
imagearr = loadarr('mnistdata/train-images.idx3-ubyte', True)
labelarr = loadarr('mnistdata/train-labels.idx1-ubyte', False)

scaler = StandardScaler().fit(imagearr)

# 加载classifier
if (os.path.isfile("mymodel.m")):
	print('从硬盘加载模型')
	clf = joblib.load("mymodel.m")
else:
	print('重新训练模型')
	
	clf = svm.SVC()
	
	t1 = time.time()
	clf.fit(imagearr[::1], labelarr[::1])
	joblib.dump(clf, "mymodel.m")
	t2 = time.time()
	print(t2 - t1, '秒')
	
file_path = sys.argv[1]
file_image = Image.open(file_path)

file_image = centerimg.rectimg(file_image)
file_image.thumbnail((28,28))
file_image = file_image.convert('L')

#file_image.show()

data = np.array(file_image.getdata()).reshape(-1, 28*28)

#data = scaler.transform(data)

p = clf.predict(data) 
print(p)