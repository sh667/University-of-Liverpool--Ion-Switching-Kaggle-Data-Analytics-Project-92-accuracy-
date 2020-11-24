import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
  for filename in filenames:
  print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import os

from numba import jit, cuda
os.getcwd()
train = train=pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")
train.shape
train.head()
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
test.shape
test.head()
test['time'] = (test['time'] - 500).round(4)
test.head()
train['signal'].describe()

def add_batch(data, batch_size):
  c = 'batch_' + str(batch_size)
data[c] = 0
ci = data.columns.get_loc(c)
n = int(data.shape[0] / batch_size)
print('Batch size:', batch_size, 'Column name:', c, 'Number of batches:', n)
for i in range(0, n):
  data.iloc[i * batch_size: batch_size * (i + 1), ci] = i

for batch_size in [500000, 400000, 200000,100000]:
  add_batch(train, batch_size)
add_batch(test, batch_size)

train.head()
original_batch_column = 'batch_500000'

batch_columns = [c for c in train.columns if c.startswith('batch')]
batch_columns
batch_6 = train[train[original_batch_column] == 6]

def add_shifted_signal(data, shift):
  for batch in data[original_batch_column].unique():
  m = data[original_batch_column] == batch
new_feature = 'shifted_signal_'
if shift > 0:
  shifted_signal = np.concatenate((np.zeros(shift), data.loc[m, 'signal'].values[:-shift]))
new_feature += str(shift)
else:
  t = -shift
shifted_signal = np.concatenate((data.loc[m, 'signal'].values[t:], np.zeros(t)))
new_feature += 'minus_' + str(t)
data.loc[m, new_feature] = shifted_signal

add_shifted_signal(train, -1)
add_shifted_signal(test, -1)

add_shifted_signal(train, 1)
add_shifted_signal(test, 1)

train.head()


y_train = train['open_channels'].copy()
x_train = train.drop(['time', 'open_channels'] + batch_columns, axis=1)

x_test = test.drop(['time'] + batch_columns, axis=1)

list(x_train.columns)

del train
del test

set(x_train.columns) ^ set(x_test.columns)
set()
from sklearn.preprocessing import StandardScaler
x_train = x_train.values
x_test = x_test.values
 #KNN
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train,test_size=0.20)
from sklearn.neighbors import KNeighborsRegressor
# Will take some time
for i in range(1,20):
  print('training for k=',i)
knn = KNeighborsRegressor(n_neighbors=i,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train,Y_train)
pred_i = knn.predict(X_test)
error_rate.append(np.mean(pred_i-Y_test)**2)
#Model Building and predictions
knn = KNeighborsRegressor(n_neighbors=4,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train,Y_train)
pred_i = knn.predict(x_test)
y_pred = np.round(pred_i)
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
submission['open_channels'] = pd.Series(y_pred, dtype='int32')
submission['open_channels']=submission['open_channels'].astype('int')
submission.to_csv('submission_knn04.csv', index=False, float_format='%.4f')
submission.head()

