# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:50:38 2019

@author: Pierre
"""
#########################################################################
# DetectSignificatnWebTrafficPython
# Détection  de trafic Web significatif avec Python
# Auteur : Pierre Rouarch 2019
# Données : Issues de l'API de Google Analytics - 
# Comme illustration Nous allons travailler sur les données du site 
# https://www.networking-morbihan.com 
# Site de l'association Networking Morbihan :
# https://github.com/Anakeyn/DetectSignificantWebTrafficPython/raw/master/dfPageViews.zip
#.
#############################################################
# On démarre ici !!!!
#############################################################
#def main():   #on ne va pas utiliser le main car on reste dans Spyder
#Chargement des bibliothèques utiles
import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
import scipy as sp  #pour l'analyse statistique
import pandas as pd  #pour les Dataframes ou tableaux de données
import seaborn as sns #graphiques étendues
import math #notamment pour sqrt()
from datetime import timedelta
from scipy import stats
#pip install scikit-misc  #pas d'install conda ???
from skmisc import loess  #pour methode Loess compatible avec stat_smooth
#conda install -c conda-forge plotnine
from plotnine import *  #pour ggplot like
#conda install -c conda-forge mizani 
from mizani.breaks import date_breaks  #pour personnaliser les dates affichées

#Changement du répertoire par défaut pour mettre les fichiers de sauvegarde
#dans le même répertoire que le script.
import os
print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/CHEMIN"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif

###############################################################################
#Récupération du fichier de données
###############################################################################
myDateToParse = ['date']  #pour parser la variable date en datetime sinon object
dfPageViews = pd.read_csv("dfPageViews.csv", sep=";", dtype={'Année':object}, parse_dates=myDateToParse)
#verifs
dfPageViews.dtypes
dfPageViews.count()  #72821 enregistrements 
dfPageViews.head(20)
##############################################################################
#creation de la dataframe daily_data par jour
dfDatePV = dfPageViews[['date', 'pageviews']].copy() #nouveau dataframe avec que la date et le nombre de pages vues
daily_data = dfDatePV.groupby(dfDatePV['date']).count() #
#dans l'opération précédente la date est partie dans l'index
daily_data['date'] = daily_data.index #recrée la colonne date.
daily_data['cnt_ma30'] =  daily_data['pageviews'].rolling(window=30).mean()
daily_data['Année'] = daily_data['date'].astype(str).str[:4]
daily_data['DayOfYear'] = daily_data['date'].dt.dayofyear #récupère la date du jour
daily_data.reset_index(inplace=True, drop=True)  #on reindexe 

#Graphique Moyenne Mobile 30 jours.
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.lineplot(x='DayOfYear', y='cnt_ma30', hue='Année', data= daily_data,  
                  palette=sns.color_palette("husl",n_colors=8))
fig.suptitle("Les données présentent une saisonnalité : ", fontsize=14, fontweight='bold')
ax.set(xlabel="Numéro de Jour dans l'année", ylabel='Nbre pages vues / jour en moyenne mobile',
       title="Le trafic baisse en général en été.")
fig.text(.9,-.05,"Comparatif Nbre pages vues par jour  par an moy. mob. 30 jours \n Données nettoyées", 
         fontsize=9, ha="right")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
fig.savefig("PV-Comparatif-mm30.png", bbox_inches="tight", dpi=600)

# Sauvegarde de DailyData en csv  pourra servir dans d'autres articles.
daily_data.to_csv("DailyDataCleanPython.csv", sep=";", index=False)  #séparateur ; 


##########################################################################
# Détection des événements significatifs - Données_aberrantes
# on va utiliser la méthode du Test de Tau de Thomson Modifié
# Voir ici https://fr.wikipedia.org/wiki/Donn%C3%A9e_aberrante
##########################################################################
#Etape 1 Calcul du Seuil
n=daily_data.shape[0] #taille de l'échantilon 2658
#Récupérons la valeur de Tau sur la table comme nous avons n = 2658 
#notre valeur de tau est 1,96
tau=1.96
#calculons le seuil de base
threshold = (tau*(n-1))/( math.sqrt(n) * math.sqrt(n-2+(math.pow(tau,2))) )
#threshold=1.9585842166773806

#Etape 2 Evaluation du zcore par rapport au seuil
# ici z_score = (daily_data['pageviews'] - mean)/std donné 
# par zcore de scipy.stats mais que l'on aurait pu calculer. à la main
from  scipy.stats import zscore
daily_data['pageviews_zscore'] = zscore(daily_data['pageviews'])
myOutliersBase = daily_data[daily_data['pageviews_zscore'] > threshold]
len(myOutliersBase) #136 valeurs aberrantes


#Finalement on va augmenter le seuil de façon empirique pour réduire le  
#nombre de valeurs aberrantes à un même niveau de ce que l'on avait avec R
threshold = 2.29
myOutliers = daily_data[daily_data['pageviews_zscore'] > threshold]
len(myOutliers)  #97 valeurs 



#Graphique Pages vues
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.lineplot(x='date', y='pageviews', data= daily_data)
sns.scatterplot(x='date', y='pageviews', data= myOutliers, color='red')
fig.suptitle( str(len(myOutliers)) + " événements ont été détectés :  ", fontsize=14, fontweight='bold')
ax.set(xlabel="Date", ylabel='Nbre pages vues / jour',
       title="Il y a moins d'événements significatifs les dernières années")
fig.text(.9,-.05,"Evénements significatifs depuis 2011 détectés par calcul des valeurs aberrantes", 
         fontsize=9, ha="right")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
fig.savefig("Anoms-Pageviews-s2011.png", bbox_inches="tight", dpi=600)

#Affichage sur la courbe des moyennes mobiles sur 30 jours
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.lineplot(x='date', y='cnt_ma30', data= daily_data)
sns.scatterplot(x='date', y='cnt_ma30', data= myOutliers, color='red')
fig.suptitle( str(len(myOutliers)) + " événements ont été détectés :  ", fontsize=14, fontweight='bold')
ax.set(xlabel="Date", ylabel='Nbre pages vues en moyenne mobile / jour',
       title="Il y a moins d'événements significatifs les dernières années")
fig.text(.9,-.05,"Evénements significatifs depuis 2011 détectés par calcul des valeurs aberrantes\n moyenne mobile 30 jours", 
         fontsize=9, ha="right")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()
fig.savefig("Anoms-Pageviews-s2011-mm30.png", bbox_inches="tight", dpi=600)




##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()

