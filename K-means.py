# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:34:33 2017

@author: wu
"""

import numpy
import matplotlib.pyplot 

def main():
    dataset=[]
    print("step1:loading data...")
    datasetfile=open(input("Enter the file path: "),'r')#raw clusting data
    for line in datasetfile:
        samplelist=line.strip().split()
        samplelist=map(float,samplelist)#str to float
        dataset.append(samplelist)
    dataset=numpy.mat(dataset)
    global num,dim,k
    num,dim=dataset.shape
    k=int(input("Enter the number of cluster: "))
    maxiter=int(input("Enter the maximum number of iterations: "))
    print("step2:clusting...")
    centers_result,clusterassignment_result=kmeans(dataset,maxiter)
    print("step3:show the clusting result...")
    if dim==2:
        showcluster(dataset,centers_result,clusterassignment_result)
    else:
        print("The dimension of data is too large to plot")

def kmeans(dataset,maxiter):
    itercount=0
    clusterassignment=numpy.zeros(num)
    clusterchange=True  
    centers=initcenters(dataset)
    while clusterchange and itercount<maxiter:
        clusterchange=False
        itercount+=1
        for i in range(num):        
            dis2cen=distance2centers(dataset[i],centers)
            minindex=numpy.argmin(dis2cen)
            if clusterassignment[i]!=minindex:
                clusterchange=True
                clusterassignment[i]=minindex
        for j in range(k):
            pointsincenterk=dataset[numpy.nonzero(clusterassignment==j)]
            centers[j,:]=numpy.mean(pointsincenterk,axis=0)
    return centers,clusterassignment

def initcenters(dataset):
    centers=numpy.zeros((k,dim))
    for i in range(k):
        index=numpy.random.randint(0,num)
        centers[i]=dataset[index]
    print(centers)
    return centers

def distance2centers(sample,centers):
    dis2cen=numpy.zeros(k)
    for i in range(k):
        dis2cen[i]=numpy.sqrt(numpy.sum(numpy.square(sample-centers[i,:])))
    return dis2cen

def showcluster(dataset,centers,clusterassignment):
    mark=['or','ob','og','om','oc','ok','ow','oy'] 
    for i in range(num):
        markindex=int(clusterassignment[i])
        matplotlib.pyplot.plot(dataset[i,0],dataset[i,1],mark[markindex])
    for i in range(k):
        matplotlib.pyplot.plot(centers[i,0],centers[i,1],mark[i],markersize=17)
    matplotlib.pyplot.show()

if __name__=="__main__":
    main()