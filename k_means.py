#Importing Libraries
import pandas as pd
import random
import numpy as np

#Loading dataset
dataset=pd.read_csv('Mall_Customers.csv')
data=dataset.iloc[:,[3,4]].values

#K_means class
class k_means:
    #Initialization in the class
    def __init__(self):
        self.token=0
        self.cost_init=99
        self.cost_latest=999

    #Random initializations of centroids depending on k
    def init_centroids(self,k):
        init_centros=[]
        for i in range(0,k):
            init_centros.append(data[random.randint(0,199)])
            if i!=0:
                for kal in init_centros:
                    while (init_centros[i][0]==kal[0] and init_centros[i][1]==kal[1]):
                        init_centros[i]=data[random.randint(0,199)]
        return init_centros

    #Euclidean distance
    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    #Class Assignment for each data point
    def separate_classes(self,token,k):
        label=[]
        cost=0

        if token==0:
            init_centros=self.init_centroids(k)
            token=token+1

        else:
            init_centros=self.mean_centroid(k)
            token=token+1

        for i in range(0,len(init_centros)):
            label.append([])

        for i in data:
            distances=[]
            for centre in init_centros:
                distance=self.euclidean_distance(i,centre)
                distances.append(distance)
            min_dist=min(distances)
            cost=cost+min_dist

            if min_dist!=0.0:
                min_index=distances.index(min_dist)
                label[min_index].append(i)

        cost=(1/(len(data)-2))*cost
        return label,cost


    #New_centroid calculation
    def mean_centroid(self,k):
        means=[]
        label,cost_latest=self.separate_classes(self.token,k)
        for i in range(0,len(label)):
            means.append(np.mean(label[i],axis=0))
        return means,cost_latest

    #Cost reduction for a particular k 
    def cost_computation(self,k):
        means,cost_latest=self.mean_centroid(k)
        if self.token==1:
            self.cost_init=999
        while self.cost_init-cost_latest > 0.000001:
            self.cost_init=cost_latest
            means,cost_latest=self.mean_centroid(k)
        return cost_latest

    #Repetition of whole process for a k iteratively to avoid local optima
    def iterative_init(self,k):
        all_costs=[]
        for i in range(0,100):
            cost_last=self.cost_computation(k)
            all_costs.append(cost_last)
        min_cost=min(all_costs)
        return min_cost

    #Finding minimum cost for all k to find number of clusters
    def for_ks(self):
        k_cost=[]
        for k in range(2,11):
            costing=self.iterative_init(k)
            k_cost.append(costing)
        return k_cost

p1=k_means()
k_cost=p1.for_ks()
print(k_cost)
print(p1.iterative_init(5))






