import numpy as np
import time
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier

max_total_score = 0
max_total_score_clf_name = ''
tested_clf = []

def clf_generator(common_iter = True, svc_iter = True, random_forest_iter = True):
	
	if (common_iter):
		clfs = (\
			{'name':'svc-rbf','clf':svm.SVC(), 'scale': False},\
			{'name':'svc-rbf-scaled','clf':svm.SVC(), 'scale': True},\
			{'name':'svc-linear-kernel', 'clf':svm.SVC(kernel='linear'), 'scale': True},\
			# 训练时间很久
			{'name':'linearsvc', 'clf':svm.LinearSVC(max_iter=10000), 'scale': True},\
			# 测试时间久
			{'name':'KNeighborsClassifier', 'clf':KNeighborsClassifier(), 'scale': False},\
			# 测试时间久
			{'name':'KNeighborsClassifierScale', 'clf':KNeighborsClassifier(), 'scale': True},\
			# 使用未scale的数据训练时间久
			{'name':'LogisticRegression', 'clf':LogisticRegression(max_iter=10000), 'scale': False},\
			{'name':'LogisticRegressionScale', 'clf':LogisticRegression(max_iter=10000), 'scale': True},\
			{'name':'RandomForestClassifier', 'clf':RandomForestClassifier(), 'scale': False},\
			{'name':'RandomForestClassifierScale', 'clf':RandomForestClassifier(), 'scale': True},\
			{'name':'DecisionTreeClassifier', 'clf':tree.DecisionTreeClassifier(), 'scale': False},\
			{'name':'DecisionTreeClassifierScaler', 'clf':tree.DecisionTreeClassifier(), 'scale': True},\
			#{'name':'GradientBoostingClassifier', 'clf':GradientBoostingClassifier(), 'scale': False},\
			#{'name':'GradientBoostingClassifierScaler', 'clf':GradientBoostingClassifier(), 'scale': True},\
			#{'name':'MLPClassifier', 'clf': MLPClassifier(), 'scale': False},\
			#{'name':'MLPClassifierScaled', 'clf': MLPClassifier(), 'scale': True},\
		)
		for clf_item in clfs:
			yield clf_item
		
	
	# 尝试试用SVC不同的C参数进行测试
	if (svc_iter):
		for cp in np.arange(0.1, 1, 0.1):
			clf = svm.SVC(C=cp)
			clf_name = 'svc(c=%f)' % cp
			yield {'name': clf_name, 'clf': clf, 'scale': False}
	
		for cp in np.arange(1, 10, 2):
			clf = svm.SVC(C=cp)
			clf_name = 'svc(c=%f)' % cp
			yield {'name': clf_name, 'clf': clf, 'scale': False}
	
	
	if (random_forest_iter):
		# 随机森林参数调优 （主要有3个参数，max_features, n_estimators, min_sample_leaf）
		# 最小叶子节点
		min_sample_leaf_range = range(1, 50, 10)
		# 决策树个数
		n_estimators_range = range(1, 1000, 150)
		for msl in min_sample_leaf_range:
			for ne in n_estimators_range:
				clf = RandomForestClassifier(min_samples_leaf=msl, n_estimators=ne, random_state=33)
				clf_name = 'RandomForest(min_samples_leaf=%d,n_estimators=%d)' % (msl, ne)
				yield {'name': clf_name, 'clf': clf, 'scale': False}


def cross_score (train_data, label_data, n_splits=5, clf_iter=clf_generator()):
	
	global max_total_score
	global max_total_score_clf_name
	global tested_clf
	max_total_score = 0
	max_total_score_clf_name = ''
	tested_clf = []
	
	scaler = StandardScaler().fit(train_data)
	train_data_scaled = scaler.transform(train_data)
	
	score_file = open('scorefile.txt', 'w')
	
	for clfitem in clf_iter:
		clf = clfitem['clf']
		train_data_ref = train_data
		if (clfitem.get('scale')):
			train_data_ref = train_data_scaled
			
		__test (clfitem['name'], clf, train_data_ref, label_data, score_file, n_splits)
		
	print('最优clf:%s, score_total:%f' % (max_total_score_clf_name, max_total_score))
	
	print('已测试:')
	tested_clf.sort(key = lambda x: x['total_score'], reverse = True)
	for tested_item in tested_clf:
		print(tested_item)
		
def __test (clf_name, clf, train_data, label_data, score_file, n_splits=5):
	print('-' * 30)
	print('clf:', clf_name)
	c = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=1)
	time1 = time.time()
	score = cross_val_score (clf, train_data, label_data, cv=c)
	time2 = time.time()
	print(score)
	score_total = sum(score)
	print('score total:', score_total)
	global max_total_score
	global max_total_score_clf_name
	global tested_clf
	if max_total_score < score_total:
		max_total_score = score_total
		max_total_score_clf_name = clf_name
		
	tested_clf.append({'name': clf_name, 'total_score': score_total})
	
	time_used = time2 - time1
	print('测试用时:', time_used)
	print('-' * 30)
	
	# 写文件
	score_file.write(clf_name + '\t')
	for s in score:
			score_file.write (str(s)+'\t')
	score_file.write(str(score_total) + '\t')
	score_file.write(str(time_used) + '\t')
	score_file.write('\r\n')
	score_file.flush()
	

