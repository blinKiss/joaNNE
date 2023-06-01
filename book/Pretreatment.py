import pandas as pd
class2 = pd.read_csv('./joaNNE/book/chap10/data/class2.csv')

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2['class2'])
print(train_x)