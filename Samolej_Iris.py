import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans




iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# two attributes:
x2 = iris.iloc[:, [0, 1]].values
# three attributes:
x3 = iris.iloc[:, [0, 1, 2]].values
# four attributes:
x4 = iris.iloc[:, [0, 1, 2, 3]].values

#iris.info()
#print(iris)

iris_outcome = pd.crosstab(index=iris["species"], columns="count")
print(iris_outcome)

iris_setosaOrg=50
iris_versicolorOrg=50
iris_virginicaOrg=50

clusters = 3


#--------------------------------------------------------------histograms

sns.FacetGrid(iris,hue="species",height=3).map(sns.histplot,"petal length").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.histplot,"petal width").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.histplot,"sepal length").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.histplot,"sepal width").add_legend()
plt.show()

#--------------------------------------------------------------3a

print("------------------------clustering with two attributes:")



kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x2)
iris_setosax2=len(sorted([i for i in y_kmeans if i == 0]))
iris_versicolorx2=len(sorted([i for i in y_kmeans if i == 1]))
iris_virginicax2=len(sorted([i for i in y_kmeans if i == 2]))


print('KMeans for two attributes:')
print('Iris-setosa: ', iris_setosax2)
print('Iris-versicolor: ', iris_virginicax2)
print('Iris-virginica: ', iris_versicolorx2)
print('y_kmeans.cluster_centers_')
print(y_kmeans.cluster_centers_)


if (iris_setosaOrg > iris_setosax2):
    ax2 = iris_setosaOrg - iris_setosax2
else: ax2 =0
if (iris_versicolorOrg > iris_versicolorx2):
    bx2 = iris_versicolorOrg - iris_versicolorx2
else: bx2 = 0
if (iris_virginicaOrg > iris_virginicax2):
    cx2 =iris_virginicaOrg - iris_virginicax2
else: cx2 = 0
perx2 = ax2+bx2+cx2

print("percentage of objects that ended up in a different group:", perx2/149*100 , "%")


#2D
plt.scatter(x2[y_kmeans == 0, 0], x2[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Iris setosa')
plt.scatter(x2[y_kmeans == 1, 0], x2[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Iris versicolour')
plt.scatter(x2[y_kmeans == 2, 0], x2[y_kmeans == 2, 1], s = 100, c = 'purple', label = 'Iris virginica')
plt.xlabel("sepal lenght")
plt.ylabel("sepal width")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()
plt.show()


#---------------------------------------------------------------- 3b

print("------------------------clustering with tree attributes:")


kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x3)

iris_setosax3=len(sorted([i for i in y_kmeans if i == 0]))
iris_versicolorx3=len(sorted([i for i in y_kmeans if i == 1]))
iris_virginicax3=len(sorted([i for i in y_kmeans if i == 2]))


print('KMeans for two attributes:')
print('Iris-setosa: ', iris_setosax3)
print('Iris-versicolor: ', iris_virginicax3)
print('Iris-virginica: ', iris_versicolorx3)
print(y_kmeans.cluster_centers_)

if (iris_setosaOrg > iris_setosax3):
    ax3 = iris_setosaOrg - iris_setosax3
else: ax3 =0
if (iris_versicolorOrg > iris_versicolorx3):
    bx3 = iris_versicolorOrg - iris_versicolorx3
else: bx3 = 0
if (iris_virginicaOrg > iris_virginicax3):
    cx3 =iris_virginicaOrg - iris_virginicax3
else: cx3 = 0
perx3 = ax3+bx3+cx3

print("percentage of objects that ended up in a different group:", perx3/149*100 , "%")

#3D
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x3[y_kmeans == 0, 0], x3[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Iris setosa')
plt.scatter(x3[y_kmeans == 1, 0], x3[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Iris versicolour')
plt.scatter(x3[y_kmeans == 2, 0], x3[y_kmeans == 2, 1], s = 100, c = 'purple', label = 'Iris virginica')
ax.set_xlabel(r"sepal lenght", fontsize=15, rotation=60)
ax.set_ylabel("sepal width", fontsize=15, rotation=60)
ax.set_zlabel("Petal Length", fontsize=15, rotation=60)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.show()



#--------------------------------------------------------------3c

print("------------------------clustering with four attributes:")

kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x4)

iris_setosax4=len(sorted([i for i in y_kmeans if i == 0]))
iris_versicolorx4=len(sorted([i for i in y_kmeans if i == 1]))
iris_virginicax4=len(sorted([i for i in y_kmeans if i == 2]))


print('KMeans for two attributes:')
print('Iris-setosa: ', iris_setosax4)
print('Iris-versicolor: ', iris_virginicax4)
print('Iris-virginica: ', iris_versicolorx4)
print(y_kmeans.cluster_centers_)

if (iris_setosaOrg > iris_setosax4):
    ax4 = iris_setosaOrg - iris_setosax4
else: ax4 =0
if (iris_versicolorOrg > iris_versicolorx4):
    bx4 = iris_versicolorOrg - iris_versicolorx4
else: bx4 = 0
if (iris_virginicaOrg > iris_virginicax4):
    cx4 =iris_virginicaOrg - iris_virginicax4
else: cx4 = 0
perx4 = ax4+bx4+cx4

print("percentage of objects that ended up in a different group:", perx4/149*100 , "%")

