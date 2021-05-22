import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from datetime import datetime

train = pd.read_csv("train.csv")
testA = pd.read_csv("test.csv")
resultTure = pd.read_csv("result.csv")

# 将表格中非字符串的列选出来，并作为一个list
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
numerical_fea.remove('isDefault')

# 中位数填充缺失值
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
testA[numerical_fea] = testA[numerical_fea].fillna(testA[numerical_fea].median())

# str 类型映射到num trian test 应该保持相同
for data in [train]:
    data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
    startDate = datetime.strptime('2007-06-01', '%Y-%m-%d')
    data['issueDate'] = data['issueDate'].apply(lambda x: x-startDate).dt.days
    data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    data['employmentLength'] = data['employmentLength'].map(
    {'1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
     '9 years': 9, '10+ years': 10, '< 1 year': 0})
    data['subGrade'] = data['subGrade'].map(
    {'E2': 1, 'D2': 2, 'D3': 3, 'A4': 4, 'C2': 5, 'A5': 6, 'C3': 7, 'B4': 8, 'B5': 9, 'E5': 10,
     'D4': 11, 'B3': 12, 'B2': 13, 'D1': 14, 'E1': 15, 'C5': 16, 'C1': 17, 'A2': 18, 'A3': 19, 'B1': 20,
     'E3': 21, 'F1': 22, 'C4': 23, 'A1': 24, 'D5': 25, 'F2': 26, 'E4': 27, 'F3': 28, 'G2': 29, 'F5': 30,
     'G3': 31, 'G1': 32, 'F4': 33, 'G4': 34, 'G5': 35})
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
#  data['n15']=data['n8']*data['n10']

for data in [testA]:
    data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
    startDate = datetime.strptime('2007-06-01', '%Y-%m-%d')
    data['issueDate'] = data['issueDate'].apply(lambda x: x-startDate).dt.days
    data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    data['employmentLength'] = data['employmentLength'].map(
        {'1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8,
         '9 years': 9, '10+ years': 10, '< 1 year': 0})
    data['subGrade'] = data['subGrade'].map(
        {'E2': 1, 'D2': 2, 'D3': 3, 'A4': 4, 'C2': 5, 'A5': 6, 'C3': 7, 'B4': 8, 'B5': 9, 'E5': 10,
         'D4': 11, 'B3': 12, 'B2': 13, 'D1': 14, 'E1': 15, 'C5': 16, 'C1': 17, 'A2': 18, 'A3': 19, 'B1': 20,
         'E3': 21, 'F1': 22, 'C4': 23, 'A1': 24, 'D5': 25, 'F2': 26, 'E4': 27, 'F3': 28, 'G2': 29, 'F5': 30,
         'G3': 31, 'G1': 32, 'F4': 33, 'G4': 34, 'G5': 35})
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

print("数据预处理完成!")

# 划分训练集和验证集
sub = testA[['id']].copy()
sub['isDefault'] = 0
testA = testA.drop(['id', 'issueDate'], axis=1)
data_x = train.drop(['isDefault', 'id', 'issueDate'], axis=1)
data_y = train[['isDefault']].copy()
x, val_x, y, val_y = train_test_split(
    data_x,
    data_y,
    test_size=0.428,
    random_state=2,
    stratify=data_y
)

col = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode',
       'regionCode',
       'initialListStatus', 'applicationType', 'policyCode']
for i in data_x.columns:
    if i in col:
        data_x[i] = data_x[i].astype('str')
for i in testA.columns:
    if i in col:
        testA[i] = testA[i].astype('str')

# clf = lgb.LGBMClassifier()

# 定义模型 catboost
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="CPU",
    learning_rate=0.1,
    iterations=100,
    random_seed=5020,
    od_type="Iter",
    depth=10)

answers = []
mean_score = 0
n_folds = 5
# 通过5折交叉验证 训练模型
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2021)
for train, test in sk.split(data_x, data_y):
    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]
    clf = model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=10, cat_features=col)
    yy_pred_valid = clf.predict(x_test, prediction_type='Probability')[:, -1]
    # print('catboost validation auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))
    mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds  # 计算AUC
    y_pred_valid = clf.predict(testA, prediction_type='Probability')[:, -1]
    answers.append(y_pred_valid)
# print('mean valAuc:{}'.format(mean_score))
cat_pre = sum(answers) / n_folds
sub['isDefault'] = cat_pre
# 生成提交的文件
sub.to_csv('submit.csv', index=False)

# submit = pd.read_csv("submit.csv")
resultTure = pd.read_csv("result.csv")
submit = sub.sort_values('id', ascending=True)
resultTure = resultTure.sort_values('id', ascending=True)
data_y = resultTure[['isDefault']].copy()
data_y_predict = submit[['isDefault']].copy()
print()
print('result vs submit auc:{}'.format(roc_auc_score(data_y, data_y_predict)))
