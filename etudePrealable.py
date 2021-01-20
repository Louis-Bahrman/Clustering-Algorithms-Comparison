#%% Imports

import pandas as pd
import numpy as np

import scipy.cluster.hierarchy as sch
import sklearn.cluster as skc
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import sklearn.mixture as mix
import sklearn.decomposition as skd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from time import time
from functools import wraps


#%% Fonctions d'initialisation et de pré-traitement

def importTestData(filepath):
    df=pd.read_csv(filepath,delimiter='\s+')
    avecResultats=df.shape[1]==3
    if avecResultats:
        df.columns=['c1','c2','resultat']
        resultats=df['resultat']
        df=df.drop('resultat',axis=1)
    data=centrerReduire(df)
    if avecResultats:
        return data, resultats
    else:
        return data

def centrerReduire(df):
    data=df.to_numpy()
    scaleur=skp.StandardScaler()
    data=scaleur.fit_transform(data)
    return data

#%% Mesures et exploitation

def mesureTps(fonction):
    @wraps(fonction)
    def wrapper(*args, **kwargs):
        start=time()
        res=fonction(*args,**kwargs)
        end=time()
        t=end-start
        if t>1:
            print("temps : ",t)
        return res
    return wrapper

def tailleClusters(prediction):
    """Série du nombre d'elements par cluster"""
    return pd.Series(prediction).value_counts()

def tailleClusterBruit(prediction):
    """Utile uniquement avec le dbscan"""
    taille=tailleClusters(prediction)
    if -1 in taille:
        return taille[-1]
    else:
        return 0
    
def nombreClusters(prediction):
    return len(np.unique(prediction))
    

def score(prediction, data=None, vraiesValeurs=None):
    tc=tailleClusters(prediction)
    print('Taille des clusters : ', tc)
    if (vraiesValeurs is None and data is not None):
        s=skm.silhouette_score(data, prediction)
        print('silhouette-score : ', s)
    if (vraiesValeurs is not None):
        s=skm.adjusted_rand_score(vraiesValeurs, prediction)
        print('rand : ',s)
    return s

#%% Algorithmes

@mesureTps
def kmeans(data,k,  algo='elkan', plot=False, dimensionPlot=2):
    km=skc.KMeans(k,init='k-means++',algorithm=algo,n_init=100)
    km.fit(data)
    predict=km.predict(data)
    if plot:
        centers = km.cluster_centers_
        affiche(data,predict,centers=centers, dimension=dimensionPlot)
    return predict


@mesureTps
def gaussian(data,k, plot=False, dimensionPlot=2):
    gm=mix.GaussianMixture(n_components=k,n_init=100)
    gm.fit(data)
    predict=gm.predict(data)
    return predict

@mesureTps
def cha(data, t, z=None, methode='ward', metrique='euclidean', plot=False,dimensionPlot=2):
    if z is None:
        z=sch.linkage(data, method=methode , metric=metrique, optimal_ordering=plot)
    predict=sch.fcluster(z, t, criterion='distance')
    if plot:
        sch.dendrogram(z)
        affiche(data,predict, dimension=dimensionPlot)
    return predict

@mesureTps
def dbscan(data, epsilon, nVoisins=5, metrique='minkowski',p=2, plot=False,dimensionPlot=2):    
    predict=skc.dbscan(data,eps=epsilon,  min_samples=nVoisins, metric=metrique, p=2)[1]
    if plot:
        affiche(data,predict, dimension=dimensionPlot)
    return predict


#%% Affichage

def affiche(data, resultat=None, dimension=2, centers=None):
    """
        affichage 2D ou 3D
    """
    fig = plt.figure()   
    if data.shape[1]>=3 and dimension==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Axe 1')
        ax.set_ylabel('Axe 2')
        ax.set_zlabel('Axe 3')
        ax.scatter(data[:,0],data[:,1],data[:,2], c=resultat)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:2], c='black', s=200, alpha=0.5);
    else:
        ax=fig.add_subplot(111)
        ax.scatter(data[:,0],data[:,1],c=resultat)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    fig.show()

def afficheFrame(df,colonneAbscisses,colonneOrdonnee, prediction=None):
    plt.scatter(df[colonneAbscisses], df[colonneOrdonnee], c=prediction)
    plt.xlabel(colonneAbscisses)
    plt.ylabel(colonneOrdonnee)
    plt.show()

def afficheFrameComplete(df, prediction=None):
    for i in range(len(df.columns)):
        for j in range(i):
           afficheFrame(df, df.columns[i], df.columns[j], prediction)


#%% Appels aux fonctions

if __name__=="__main__":
    dataAggregation,resultatsAggregation=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/Aggregation.txt')
    dataG2_2_20=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/g2-2-20.txt')
    dataG2_2_100=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/g2-2-100.txt')
    dataJain,resultatsJain=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/jain.txt')
    dataPathbased,resultatsPathbased=importTestData('/home/louis/Documents/IMT/ODATA/Projet/Donnees_projet_2020/pathbased.txt')

    