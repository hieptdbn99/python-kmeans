import random
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sklearn

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist


def main():
    # Importing the dataset
    dataset = pd.read_csv('D:\Downloads\Sales_Transactions_Dataset_Weekly.csv')
    #get column 50 to 53
    X = dataset.iloc[:, 55:58].values
    print(X)
    #declare stop condition to use in calculateNewCluster

    
    #elbow method to find the optimal number of clusters
    Kmean(len(X),X,6)
    process_test(len(X),X,6)


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



def getDistortion(listObj,k):
    n = len(listObj)
    c =[]
    for i in range(k):
        idx = random.randint(0,n-1)
        c.append(listObj[idx])
    c,flag,cluster = calculateNewCluster(c,listObj,k)
    while flag == False:
        c,flag,cluster = calculateNewCluster(c,listObj,k)
    sum_avg_dis = 0
    for i in range(k):
        avg_dis = avg_square_dis_cluster(c[i],cluster[i])
        sum_avg_dis += avg_dis
    return sum_avg_dis/k


def calculateNewCluster(c, listObj, k):
    n = len(listObj)
    distance =[]
    for i in range(k):
        temp = []
        for j in range(n):
            m = c[i]
            temp.append(L1distance_n_dim(c[i],listObj[j]))
        distance.append(temp)
    min_dis_idx = np.argmin(distance,axis=0)
    cluster =[]
    for j in range(k):
        temp = []
        for i in range(n):
            if min_dis_idx[i]==j :
                temp.append(listObj[i])
        cluster.append(temp)
    flag = True
    for i in range(k):
        ((1,2,3),(2,3,4))
        mean_val = np.mean(cluster[i],axis=0)
        if L1distance_n_dim(c[i],mean_val) > 0.05:
            c[i] = mean_val
            flag = False
    return c,flag,cluster




def writeElementCluster(listObj, pred_label, center_point, K):
    f = open('kmean_test.output', 'w')
    #print cluster
    # for i in range(len(listObj)):
    #     f.write(str(pred_label[i])+'\n')
    #print center point
    for i in range(K):
        f.write(str(center_point[i])+'\n')
    f.close()



def process_test(n,listObj,K):
    kmeans_sklearn = KMeans(n_clusters=K, random_state=0).fit(listObj)
    center_point = kmeans_sklearn.cluster_centers_
    pred_label = kmeans_sklearn.predict(listObj)
    #plt show Oxyz
    # plt.scatter(listObj[:, 0], listObj[:, 1], c=pred_label, s=50, cmap='viridis')
    # plt.scatter(center_point[:, 0], center_point[:, 1], c='black', s=200, alpha=0.5)
    # plt.show()
    plt.show()


    writeElementCluster(listObj,pred_label,center_point,K)



def avg_square_dis_cluster(c, arraysCluster):
    #return average of square distance of each point in cluster to the mean of cluster
    n = len(arraysCluster)
    sum_dis = 0
    for i in range(n):
        sum_dis += L1distance_n_dim(c, arraysCluster[i])
    return sum_dis/n

def L1distance_n_dim(pointA,pointB):
    dis = 0
    for i in range(len(pointA)):
        dis += np.abs(pointA[i]-pointB[i])
    return dis



def Kmean(n, listObj, k):
    c = []
    for i in range(k):
        idx = random.randint(0, n - 1)
        c.append(listObj[idx])
    c, flag, cluster = calculateNewCluster(c, listObj, k)
    while flag == False:
        c, flag, cluster = calculateNewCluster(c, listObj, k)
    f = open('kmean.output', 'w')
    for i in range(k):
        f.write("Center point {}:{}\n".format(i + 1, c[i]))
    f.close()
    return cluster


# run main
if __name__ == "__main__":
    main()
