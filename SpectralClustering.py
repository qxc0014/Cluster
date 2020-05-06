import sys
sys.path.append("..")
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle, islice
import numpy as np

from sklearn import datasets

def caldistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

np.random.seed(1)
#创建数据集
print('start generate datasets ...')
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
print('datasets generated over')
default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}
datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})]
#构造距离矩阵
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)
    dataset = np.array(dataset[0])
    dis = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(i+1, len(dataset)):
            dis[i][j] = 1.0 * caldistance(dataset[i], dataset[j])
            dis[j][i] = dis[i][j]
    #构造相似度矩阵
    xsize = len(dis)
    sim = np.zeros((xsize,xsize))
    for i in range(xsize):
        index = zip(dis[i], range(xsize))
        index = sorted(index, key=lambda x:x[0])
        knn_index = [index[w][1] for w in range(13)] 
        for j in knn_index:
            if j!=i:
                sim[i][j] = 10./dis[i][j]
                sim[j][i] = sim[i][j] 
    #构造拉普拉斯矩阵
    DMatrix = np.sum(sim, axis=1)
    LMatrix = np.diag(DMatrix) - sim
    #sqrtDegreeMatrix = np.diag(1.0 / (DMatrix ** (0.5)))
    #LMatrix = np.dot(np.dot(sqrtDegreeMatrix, LMatrix), sqrtDegreeMatrix)
    #svd分解
    U, V = np.linalg.eig(LMatrix)
    U = zip(U, range(len(U)))
    U = sorted(U, key=lambda U:U[0])
    H = np.vstack([V[:,i] for (v, i) in U[:n_samples]]).T
    H = H[:,0:params['n_clusters']]
    print(H)
    #kmeans
    kmeans = KMeans(n_clusters=params['n_clusters'],init='k-means++').fit(H) 
    y_pred = kmeans.labels_.astype(np.int)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                    '#f781bf', '#a65628', '#984ea3',
                                                    '#999999', '#e41a1c', '#dede00']),
                                            int(max(y_pred) + 1))))
    plt.subplot(111)
    plt.scatter(dataset[:,0], dataset[:,1], s=10, color=colors[y_pred])
    plt.title("Spectral Clustering")
    plt.show()
