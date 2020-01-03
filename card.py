import numpy as np
import time
from sklearn import svm
import joblib
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import centerimg

import cross_score

from PIL import Image

#help(Image.Image)

def gettraindata(istest=False, rotate_train=True):
	
	ret = []
	
	# 3个目录
	imgdirs = ['changka', 'fanka']
	
	# rect_i = 0

	for idx,imgdir in enumerate(imgdirs):
		#print(idx)
		#print(imgdir)
		
		trainprefix = 'card' if not istest else 'testcard'
		transformprefix = 'cardtransform' if not istest else 'testcardtransform'
		
		imgdirname = os.path.join(trainprefix, imgdir)
		imgdirtransform = os.path.join(transformprefix, imgdir)
		
		if (not os.path.exists(imgdirtransform)):
			print("创建目录", imgdirtransform)
			os.makedirs(imgdirtransform)

		for root, dirs, files in os.walk(imgdirname):
			#print(root)
			#print(files)
			
			for f in files:
				fname = os.path.join(root,f)
				ftransform = os.path.join(imgdirtransform, f)
				
				if (not os.path.exists(ftransform)):
					
					if (fname.endswith('.ini')):
						continue
					
					#print('转换文件格式')
					img = Image.open(fname)
					
					# 正方形裁剪 (输入的训练数据需相同长度，否则报错，都裁剪为正方形可保证大小相同)
					img = centerimg.rectimg(img)
					
					# 用resize缩小
					'''
					img = centerimg.rectimg(img)
					img = img.resize((100,0), Image.ANTIALIAS)
					'''
					
					# 用thumbnail缩小
					img = img.copy()
					img.thumbnail((300,300))
					
					# 保存
					img.save(ftransform)
					img = Image.open(ftransform)
					
					# 灰度图
					img = img.convert('L')
				else:
					img = Image.open(ftransform)
					
					# 灰度图
					img = img.convert('L')
					
				data = img.getdata()
				
				# reshape，保证训练数据只能是2维，大于2维会报错
				#nparray = np.array(data, np.uint8).reshape(-1, 3)
				nparray = np.array(data, np.uint8).reshape(-1)
				
				print(nparray.shape)
				
				ret.append(nparray)
				
				# 如果旋转标记位True，则加入原图像其他3个方向的旋转变换
				if (rotate_train):
					for rotate_i in range(1,4):
						deg = rotate_i * 90
						img_rotate = img.rotate(deg)
						data = img_rotate.getdata()
						nparray = np.array(data, np.uint8).reshape(-1)
						ret.append(nparray)
				
	return np.array(ret)
	
def getlabeldata(istest=False, rotate_train=True):
	
	ret = []
	
	# 3个目录
	imgdirs = ['changka', 'fanka']

	for idx,imgdir in enumerate(imgdirs) :
		
		prefix = 'card' if not istest else 'testcard'
		
		imgdirname = os.path.join(prefix, imgdir)

		for root, dirs, files in os.walk(imgdirname):
			for f in files:
				ret.append(idx)
				if (rotate_train):
					ret.append(idx)
					ret.append(idx)
					ret.append(idx)
	
	return np.array(ret)


rotate_train_flag = False

traindata = gettraindata(rotate_train=rotate_train_flag)
labeldata = getlabeldata(rotate_train=rotate_train_flag)


# scaler
#scaler = StandardScaler().fit(traindata)
#traindata = scaler.transform(traindata)

# 测试testcard文件夹中的数据
clf = RandomForestClassifier()
print(clf)
# clf = svm.SVC()
clf.fit(traindata, labeldata)
# 测试数据
test_data = gettraindata(istest=True,rotate_train=rotate_train_flag)
test_label = getlabeldata(istest=True,rotate_train=rotate_train_flag)
score = clf.score(test_data, test_label)
print(score)


# 交叉测试
cross_score.cross_score (traindata, labeldata, 8, clf_iter = cross_score.clf_generator(common_iter = True, svc_iter = False, random_forest_iter=False))

'''
x_train, x_test, y_train, y_test = train_test_split(traindata, labeldata, test_size=0.2, random_state=7)

clf = svm.SVC()
clf.fit(x_train, y_train)

p = clf.predict(x_test)
print(p)

s = clf.score(x_test, y_test)
print(s)
'''

