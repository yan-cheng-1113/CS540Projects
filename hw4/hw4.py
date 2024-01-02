import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.cluster.hierarchy as scipy

def load_data(filepath):
    dictList = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dictList.append(row)
    return dictList

def calc_features(row):
    a = np.zeros((6, ), dtype=int)
    a[0] = row['Attack']
    a[1] = row['Sp. Atk']
    a[2] = row['Speed']
    a[3] = row['Defense']
    a[4] = row['Sp. Def']
    a[5] = row['HP']
    return a

#Helper method that calculates the complete distance between clusters; c1, c2 are two clusters
def calc_dist(features, c1, c2):
    D = 0
    fea_arr = np.array(features)
    n = len(fea_arr)
    d = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1, n):
            d[i,j] = np.linalg.norm(fea_arr[i, :]- fea_arr[j, :])
            d[j,i] = d[i,j]
    for i in c1:
        for j in c2:
            if(d[i][j]> D):
                D = d[i][j]
    return D

def hac(features):
    pokemon_arr = [] 
    cluster = []  
    fea_arr = np.array(features)
    n = fea_arr.shape[0]
    z = np.zeros([n-1, 4]) 

    for i in range(n):
        pokemon_arr.append([i]) 
        cluster.append(i)   
    
    
   
    for c in range(n-1):
        D_mtx = np.zeros([len(cluster),len(cluster)])
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                D_mtx[i][j] = calc_dist(features,pokemon_arr[i], pokemon_arr[j])
                D_mtx[j][i] = D_mtx[i][j]
        
        min_d = float('inf')
        min_i = 0
        min_j = 0
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                if( 0< D_mtx[i,j]< min_d):
                    min_d = D_mtx[i,j]
                    min_i = i
                    min_j = j

        cluster.append(n+c)
        cluster_j = cluster[min_j]
        cluster_i = cluster[min_i]
        
        cluster.remove(cluster[min_j])
        cluster.remove(cluster[min_i])
        
        combinedCluster = pokemon_arr[min_i] + pokemon_arr[min_j]
        pokemon_arr.append(combinedCluster)
        pokemon_arr.remove(pokemon_arr[min_j])
        pokemon_arr.remove(pokemon_arr[min_i])
        
        z[c,0] = cluster_i
        z[c,1] = cluster_j
        z[c,2] = min_d
        z[c,3] = len(pokemon_arr[n-c-2])

    return z
    
def imshow_hac(Z, names):
    plt.figure()
    dn = scipy.dendrogram(Z)
    plt.title(f'N = {len(Z) + 1}')
    plt.xlabel(names)
    plt.show()
