import pandas as pd
import lightgbm as lgb
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

submit = pd.read_csv("submit.csv")
resultTure = pd.read_csv("C:\\Users\\liuze\\Desktop\\bit\\result.csv")

submit = submit.sort_values('id',ascending = True)
resultTure = resultTure.sort_values('id',ascending = True)
# submit1.to_csv('filterOrder.csv',index = False)

data_y = resultTure[['isDefault']].copy()
data_y_predict = submit[['isDefault']].copy()
print('result vs submit auc:{}'.format(roc_auc_score(data_y,data_y_predict)))
