from nettoyage import *
import nettoyage
from score import *
import score
from fonction import *
import fonction

#Original data
df_train = pd.read_csv('/content/drive/Shareddrives/Kaggle/data/train.csv', sep=';', decimal = ',')
df_test = pd.read_csv('/content/drive/Shareddrives/Kaggle/data/test.csv', sep=';', decimal = ',')

Y_train = df_train['Points']
Id = df_test['id']

df_train = df_train.drop(['Points'], axis=1) #On supprime la colonne points mais on la conserve

#Necessaire au traitement (nettoyage.py)
l1_test, l2_test, l3_test = listes_df(df_test)
l1_train, l2_train, l3_train = listes_df(df_train) #On creer les listes
union_empty_col = Union(l3_test, l3_train)

#Pretraitement actuel (nettoyage.py)
df_test,l1_test = listes_df2(df_test,union_empty_col)
df_train,l1_train = listes_df2(df_train,union_empty_col)

#Traitement actuel (nettoyage.py)
df_train = nettoie(df_train)
df_test = nettoie(df_test)
df_train = add_moyenne(df_train)
df_test = add_moyenne(df_test)
# df_test = add_moyenne_liste(df_test,liste_s,liste_moyenne)
# df_train = add_moyenne_liste(df_train,liste_s,liste_moyenne)
# df_train = df_train.drop(['Trimestre 1 Première','Trimestre 2 Première','Trimestre 3 Première','Trimestre 3 Terminale'],axis = 1)
# df_test = df_test.drop(['Trimestre 1 Première','Trimestre 2 Première','Trimestre 3 Première','Trimestre 3 Terminale'],axis = 1)
# df_test = add_moyenne_liste(df_test,liste_matiere,liste_moy_mat)
# df_train = add_moyenne_liste(df_train,liste_matiere,liste_moy_mat)

#Prediction actuelle (fonction.py)
pred = predit

#observation des resultats (score.py)
def affiche_test(X = df_train, y = Y_train, pred = pred):
  tests = result(X,y,pred)
  mean = np.mean(tests)
  print(tests)
  print('RSME Mean: {:.2f}'.format(mean))

def affiche_res(X_tr = df_train, X_te = df_test, y = Y_train,
  pred = pred, Id = Id, export = False):
  y_pr = pred(X_tr,X_te,y)
  resultat = res(y_pr, Id)
  if export == True:
    resultat.to_csv('submission_moyenne.csv', index=False)
  return resultat