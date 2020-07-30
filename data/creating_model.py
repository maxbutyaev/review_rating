import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor

start_time = time.time()
print('Started', time.ctime())

datadir = os.path.abspath(os.curdir) + '/'

trainneg = pd.read_csv(datadir + 'trainneg.csv', sep=';', index_col='ids').drop('comments', axis=1)
trainneg['is_positive'] = [0] * len(trainneg)
trainpos = pd.read_csv(datadir + 'trainpos.csv', sep=';', index_col='ids').drop('comments', axis=1)
trainpos['is_positive'] = [1] * len(trainpos)
testneg = pd.read_csv(datadir + 'testneg.csv', sep=';', index_col='ids').drop('comments', axis=1)
testneg['is_positive'] = [0] * len(testneg)
testpos = pd.read_csv(datadir + 'testpos.csv', sep=';', index_col='ids').drop('comments', axis=1)
testpos['is_positive'] = [1] * len(testpos)
traindata = trainpos.append(trainneg).reset_index()
testdata = testpos.append(testneg).reset_index()

# max_df=0.4 deletes frequent unimportant words like 'movie', 'film', 'see' and so on
# max_features=5000 deletes rarely met words and leaves satisfctory amount of features
vectorizer = CountVectorizer(analyzer="word", max_features=5000, max_df=0.4)
traindata_features = vectorizer.fit_transform(traindata.tokened_comments).astype(np.uint8).toarray()
vocabulary = vectorizer.get_feature_names()

# save vectorizer
joblib.dump(vectorizer, datadir + 'vectorizer.json')
print('Vectorizer saved', '--- %s seconds ---' % (time.time() - start_time))
testdata_features = vectorizer.transform(testdata.tokened_comments).astype(np.uint8).toarray()

X_train = traindata_features
y_train = (traindata.marks - 1) / 9
X_test = testdata_features
y_test = (testdata.marks - 1) / 9
print('Datasets got', '--- %s seconds ---' % (time.time() - start_time))
params = {'base_score': 0.5,
          'booster': 'gbtree',
          'colsample_bylevel': 1,
          'colsample_bynode': 1,
          'colsample_bytree': 1,
          'gamma': 0,
          'importance_type': 'gain',
          'learning_rate': 0.1,
          'max_delta_step': 0,
          'max_depth': 6,
          'min_child_weight': 1,
          'n_estimators': 1450,
          'n_jobs': 1,
          'nthread': None,
          'objective': 'reg:squarederror',
          'random_state': 0,
          'reg_alpha': 0,
          'reg_lambda': 1,
          'scale_pos_weight': 1,
          'seed': None,
          'silent': None,
          'subsample': 1,
          'verbosity': 1}
rfr = XGBRegressor(**params)
rfr.fit(X_train, y_train)
print('fitted', '--- %s seconds ---' % (time.time() - start_time))
y_pred = rfr.predict(X_test)
print('R^2=', rfr.score(X_test, y_test))
print('RFR_params:', rfr.get_params())
print('Finished', time.ctime())
# save model
joblib.dump(rfr, datadir + 'JLmodel_' + \
            str(rfr.get_params()['n_estimators']) + '_' + \
            str(rfr.get_params()['max_depth']) + '.json')
