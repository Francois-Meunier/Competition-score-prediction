import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

def predit(X_tr,X_te,y):
  linreg = LinearRegression()
  linreg.fit(X_tr, y)
  # X_te['Score'] = linreg.predict(X_te)
  # X_te.at[(X_te['Score']>120) & (X_te['Bac_scientific'] == 1),'Score'] = 120
  # X_te.at[X_te['Score']<0,'Score'] = 0
  # X_te.at[X_te['Moyenne']<4,'Score'] = 0
  # y_pred = X_te['Score']
  # y_pred = y_pred.astype('int64')
  y_pred = linreg.predict(X_te)
  y_pred[y_pred <0 ] = 0
  y_pred[y_pred >145 ] = 130
  return y_pred

def predit2(X_tr,X_te,y):
  bac_s_tr,bac_sti2d_tr = separt(X_tr)
  bac_s_te,bac_sti2d_te = separt(X_te)
  linreg = LinearRegression()
  linreg.fit(bac_sti2d_tr, y[bac_sti2d_tr.index])
  y_pred = pd.DataFrame(linreg.predict(bac_sti2d_te),index = bac_sti2d_te.index)
  y_pred[y_pred >120 ] = 120
  linreg.fit(bac_s_tr, y[bac_s_tr.index])
  y_pred = pd.concat([y_pred,pd.DataFrame(linreg.predict(bac_s_te),index = bac_s_te.index)])
  y_pred = y_pred.sort_index(axis=0)
  # X_te['Score'] = linreg.predict(X_te)
  # X_te.at[(X_te['Score']>120) & (X_te['Bac_scientific'] == 1),'Score'] = 120
  # X_te.at[X_te['Score']<0,'Score'] = 0
  # X_te.at[X_te['Moyenne']<4,'Score'] = 0
  # y_pred = X_te['Score']
  # y_pred = y_pred.astype('int64')
  y_pred = y_pred[0]
  y_pred[y_pred <0 ] = 0
  y_pred[y_pred >145 ] = 130
  y_pred = y_pred.astype('int64')
  return y_pred

##gbm 4.54
def predit3(X_tr,X_te,y):
  boost = xgb.XGBRegressor(seed=42)
  boost.fit(X_tr, y)
  y_pred = boost.predict(X_te)
  y_pred[y_pred <0 ] = 0
  y_pred[y_pred >145 ] = 130
  return y_pred

# ##gbm NUL
# def predit4(X_tr,X_te,y):
#   bac_s_tr,bac_sti2d_tr = separt(X_tr)
#   bac_s_te,bac_sti2d_te = separt(X_te)
#   boost = xgb.XGBRegressor(seed=42)
#   boost.fit(bac_s_tr, y[bac_s_tr.index])
#   y_pred = pd.DataFrame(boost.predict(bac_s_te),index = bac_s_te.index)
#   y_pred[y_pred >120 ] = 120
#   boost.fit(bac_sti2d_tr, y[bac_sti2d_tr.index])
#   y_pred = pd.concat([y_pred,pd.DataFrame(boost.predict(bac_sti2d_te),index = bac_sti2d_te.index)])
#   y_pred = y_pred.sort_index(axis=0)
#   y_pred = y_pred[0]
#   y_pred[y_pred <0 ] = 0
#   # y_pred[y_pred >145 ] = 130
#   y_pred = y_pred.astype('int64')
#   return y_pred

##gbm 4.44
def predit5(X_tr,X_te,y):
  boost_ctrl = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, seed=42)
  boost_ctrl.fit(X_tr, y)
  y_pred = boost_ctrl.predict(X_te)
  y_pred[y_pred <0 ] = 0
  y_pred[y_pred >145 ] = 130
  return y_pred




def separt(X):
  bac_s = X.loc[X['Bac_scientific']==1,:]
  bac_sti2d = X.loc[X['Bac_scientific']==0,:]
  return bac_s,bac_sti2d