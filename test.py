from sklearn import datasets

#iris = datasets.load_iris()
digits = datasets.load_digits()

#print(digits)
#print(digits.images[0])
#print(digits.data[0])
#print(len(digits.data))
print(type(digits.data))
print(digits.data.shape)
print(digits.images.shape)


from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])
	
p = clf.predict(digits.data[-1:])
	
print(p)