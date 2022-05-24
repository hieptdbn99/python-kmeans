import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sklearn
from sklearn.cluster import KMeans



def main():
    # Importing the dataset
    dataset = pd.read_csv('D:\pythonProject1\Sales_Transactions_Dataset_Weekly.csv')
    X = dataset.iloc[:, 54:57].values
    X = np.array(X)

    # Elbow(9, X)
    Kmean(len(X), X, 5)
    process_test(X,5)


def Elbow(range_k, listObj):
    distortion = []
    list_k = []
    for k in range(2, range_k):
        list_k.append(k)
        dis = getDistortion(listObj, k)
        distortion.append(dis)
    # print("Distortion:============\n",distortion)
    plt.figure()
    plt.plot(list_k, distortion)
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.show()


def getDistortion(listObj, k):
    n = len(listObj)
    c = []
    for i in range(k):
        idx = random.randint(0, n - 1)
        c.append(listObj[idx])
    c, flag, cluster = calculateNewCluster(c, listObj, k)
    while flag == False:
        c, flag, cluster = calculateNewCluster(c, listObj, k)
    sum_avg_dis = 0
    for i in range(k):
        avg_dis = avg_square_dis_cluster(c[i], cluster[i])
        sum_avg_dis += avg_dis
    return sum_avg_dis / k


def calculateNewCluster(c, listObj, k):
    n = len(listObj)
    distance = []
    for i in range(k):
        temp = []
        for j in range(n):
            temp.append(L1distance_n_dim(c[i], listObj[j]))
        distance.append(temp)
    min_dis_idx = np.argmin(distance, axis=0)
    cluster = []
    for j in range(k):
        temp = []
        for i in range(n):
            if min_dis_idx[i] == j:
                temp.append(listObj[i])
        cluster.append(temp)
    flag = True
    for i in range(k):
        mean_val = np.mean(cluster[i], axis=0)
        if L1distance_n_dim(c[i], mean_val) > 0.00001:
            c[i] = mean_val
            flag = False

    return c, flag, cluster


def avg_square_dis_cluster(c, arraysCluster):
    # return average of square distance of each point in cluster to the center of cluster
    n = len(arraysCluster)
    sum_dis = 0
    for i in range(n):
        sum_dis += L1distance_n_dim(c, arraysCluster[i])
    return sum_dis / n


def L1distance_n_dim(pointA, pointB):
    dis = 0
    for i in range(len(pointA)):
        dis += np.abs(pointA[i] - pointB[i])
    return dis


def Kmean(n, listObj, k):
    c = []
    for i in range(k):
        idx = random.randint(0, n - 1)
        c.append(listObj[idx])
    c, flag, cluster = calculateNewCluster(c, listObj, k)
    while flag == False:
        c, flag, cluster = calculateNewCluster(c, listObj, k)

    color = ['brown', 'green', 'blue', 'yellow', 'orange', 'pink']



    #with 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(k):
        ax.scatter(c[i][0], c[i][1], c[i][2], c='black', s=3)
        for j in range(len(cluster[i])):
            ax.scatter(cluster[i][j][0], cluster[i][j][1], cluster[i][j][2], cmap='viridis', s=3, color=color[i], marker='s')

    
    # with 2d
    # for i in range(k):
    #     plt.scatter(c[i][0], c[i][1], c='red', s=20, marker='s')
    #     for j in range(len(cluster[i])):
    #         #show  pount= cluster[i][j] with same color random by color[]
    #         plt.scatter(cluster[i][j][0], cluster[i][j][1], c=color[i],cmap='viridis',s=3)
    # plt.show()

    f = open('kmean.output', 'w')
    for i in range(k):
        f.write("Center point {}:{}\n".format(i + 1, c[i]))

    f.write("\n")
    for i in range(k):
        f.write("Center point {}:{}\n".format(i + 1, c[i]))
        for j in range(len(cluster[i])):
            f.write("{}\n".format(cluster[i][j]))
    f.close()
    return cluster


def writeElementCluster(listObj, pred_label, center_point, K):
    f = open('kmean_test.output', 'w')
    # print cluster
    # for i in range(len(listObj)):
    #     f.write(str(pred_label[i])+'\n')
    # print center point
    for i in range(K):
        f.write(str(center_point[i]) + '\n')
    f.write("\n")
    for i in range(K):
        f.write('Center point:' + str(center_point[i]) + '\n')
        for j in range(len(listObj)):
            if pred_label[j] == i:
                f.write(str(listObj[j]) + '\n')
    f.close()


def process_test(X, K):
    X = np.array(X)
    kmeans_sklearn = KMeans(n_clusters=K, random_state=0).fit(X)
    center_point = kmeans_sklearn.cluster_centers_
    pred_label = kmeans_sklearn.predict(X)
    # show 2d
    # plt.scatter(X[:, 0], X[:, 1], c=pred_label, cmap='viridis', s=3)
    # plt.scatter(center_point[:, 0], center_point[:, 1], c='black', s=20, marker='s')
    # plt.show()

    #show 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=pred_label, cmap='viridis', s=3)
    ax.scatter(center_point[:, 0], center_point[:, 1], center_point[:, 2], c='black', s=20, marker='s')
    plt.show()

    writeElementCluster(X, pred_label, center_point, K)

    # return stop condition of loop Kmeans


# run main
if __name__ == "__main__":
    main()
