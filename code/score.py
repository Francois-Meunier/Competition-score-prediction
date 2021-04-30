import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

alea = [2,6,55,48,99]

def rsme(Y_train,Y_pred):
    return(np.sqrt((1/Y_train.shape[0])*np.sum((Y_pred - Y_train)**2)))

def create_test(X,y,i):
  X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33, random_state=i)
  return X_tr, X_te, y_tr, y_te

def result(X,y,pred,alea = alea):
  tests = []
  for nb in alea :
    X_tr, X_te, y_tr, y_te = create_test(X,y,nb)
    y_pr = pred(X_tr, X_te, y_tr)
    tests.append(rsme(y_te,y_pr))
  tests = np.array(tests)
  return tests

def res(y_pr, Id):
  pred_list = y_pr.tolist()
  resultat = pd.DataFrame(columns = ['id', 'Points'])
  resultat['id'] = Id
  resultat['Points'] = pred_list
  return resultat

def export(resultat):
  resultat.to_csv('submission.csv', index=False)