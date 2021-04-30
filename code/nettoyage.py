import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns

#from categorie import ordinal, autres, nominal, numerical, moyenne, doublon
from categorie import *

#Fonction pré nettoyage
def Union(lst1, lst2): #union de 2 listes
    final_list = list(set(lst1) | set(lst2))
    return final_list

def Inter(lst1, lst2): #intersection de 2 listes
    return list(set(lst1) & set(lst2))

def listes_df(df):                      #Retourne 3 listes en fonction du df
    df_col = df.columns.tolist()        #Liste des noms des colonnes du df
    df_nan_list = []                    #Liste pour savoir quelle colonne du df est vide
    for i in df_col :
        df_nan_list.append(df[i].isnull().all())
    title_empty_col = []                #Liste des noms des colonnes vide du df
    for i in range (len(df_nan_list)) : 
        if df_nan_list[i] == True: 
            title_empty_col.append(df_col[i])
    return(df_col, df_nan_list, title_empty_col)

def listes_df2(df,u):
    df = df.drop(u,axis=1)
    l1 = df.columns.tolist()
    return(df,l1)

#Fonction nettoyage
def nettoie(df):  #premier nettoyage


    df[ordinal[1]] = df[ordinal[1]].replace("Très satisfaisante", 3)
    df[ordinal[1]] = df[ordinal[1]].replace("Satisfaisante", 2)
    df[ordinal[1]] = df[ordinal[1]].replace("Assez satisfaisante", 1)
    df[ordinal[1]] = df[ordinal[1]].replace("Peu démontrée", 0)
    df[ordinal[1]] = df[ordinal[1]].fillna(0)
        
    df[ordinal[2]] = df[ordinal[2]].replace("Très satisfaisante", 3)
    df[ordinal[2]] = df[ordinal[2]].replace("Satisfaisante", 2)
    df[ordinal[2]] = df[ordinal[2]].replace("Assez satisfaisante", 1)
    df[ordinal[2]] = df[ordinal[2]].replace("Peu démontrée", 0)
    df[ordinal[2]] = df[ordinal[2]].fillna(0)

    df[ordinal[3]] = df[ordinal[3]].replace ("Oui", 1)
    df[ordinal[3]] = df[ordinal[3]].replace("Non", 0)
    df[ordinal[3]] = df[ordinal[3]].fillna(0)

    df[ordinal[4]] = df[ordinal[4]].replace("Très satisfaisante", 3)
    df[ordinal[4]] = df[ordinal[4]].replace("Satisfaisante", 2)
    df[ordinal[4]] = df[ordinal[4]].replace("Assez satisfaisante", 1)
    df[ordinal[4]] = df[ordinal[4]].replace("Peu démontrée", 0)
    df[ordinal[4]] = df[ordinal[4]].fillna(0)

    df[ordinal[5]] = df[ordinal[5]].replace("Très bon", 4)
    df[ordinal[5]] = df[ordinal[5]].replace("Assez bon", 3)
    df[ordinal[5]] = df[ordinal[5]].replace("Bon", 2)
    df[ordinal[5]] = df[ordinal[5]].replace("Moyen", 1)
    df[ordinal[5]] = df[ordinal[5]].replace("Faible", 0)
    df[ordinal[5]] = df[ordinal[5]].fillna(0)

    df[ordinal[6]] = df[ordinal[6]].replace("Très satisfaisante", 3)
    df[ordinal[6]] = df[ordinal[6]].replace("Satisfaisante", 2)
    df[ordinal[6]] = df[ordinal[6]].replace("Assez satisfaisante", 1)
    df[ordinal[6]] = df[ordinal[6]].replace("Peu démontrée", 0)
    df[ordinal[6]] = df[ordinal[6]].fillna(0)
    
    df["Note à l'épreuve de Oral de Français (épreuve anticipée)"] = df["Note à l'épreuve de Oral de Français (épreuve anticipée)"].fillna(0)
    df["Note à l'épreuve de Ecrit de Français (épreuve anticipée)"] = df["Note à l'épreuve de Ecrit de Français (épreuve anticipée)"].fillna(0)

    df = df.drop(autres, axis=1) #a voir

    for col in nominal:
        df[col] = df[col].apply(str)

    df = pd.get_dummies(df)
    df = df.drop(doublon,axis=1)
    df = df.fillna(0) #on met les nan à 0

    return(df)

def moyennes_g(df,s,moyenne):
    df[s]=0
    df2 = df[moyenne].cumsum(axis = 1)
    df3 = (df[moyenne]==0).sum(axis = 1) #(129 - df3)
    df2[s] = df2[s]/(len(moyenne)-1-df3)
    df[s] = df2[s].fillna(0) #score de zero pour les moyennes de zeros
    #df = df.drop(df[moyenne], axis=1) #supprime les moyennes
    #df = df.fillna(0) #on met les nan à 0
    # for moy in moyenne :
    #   df[moy] = df[moy].apply(lambda x: np.log(x) if x > 0 else x)
    return(df)

def add_moyenne(df):
  df = moyennes_g(df,"Moyenne classe",moyenne_classe)
  df = moyennes_g(df,"Moyenne candidat",moyenne_candidat)
  df['Rapport'] = df['Moyenne classe'].apply(lambda x: 1/x if x!=0 else x)
  df['Rapport'] = df['Rapport']*df['Moyenne candidat']
  df = df.drop(df[moyenne_classe],axis = 1)
  return df

def add_moyenne_liste(df,l_s,l_moyenne):
  for i in range(len(l_s)):
    df = moyennes_g(df,l_s[i],l_moyenne[i])
    df = df.drop(df[list(set(l_moyenne[i])- set([l_s[i]]))],axis = 1)
  df[df[l_s]>25] = 0
  df[df[l_s]<0] = 0
  return df
