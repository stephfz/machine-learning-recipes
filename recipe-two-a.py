from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()

#printing
print iris.feature_names #attributes
print iris.target_names #classes

print iris.data[0] #values from 1st row
print iris.target[0] #class of row 0

print all dataset
for i in range(len(iris.target)):
    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])
