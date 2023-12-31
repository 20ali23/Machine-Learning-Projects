#Kitaplıkları içe aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Veri kümesini içe aktarma
dataset = pd.read_csv("Data.csv")

#Özellik matrisi ve bağımlı değişken matrisinin belirlenmesi
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Eksik verilerin ortalama değer ile değiştirilmesi
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Kategorik verilerin kodlanması
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Bağımlı değişken matrisinin kodlanması
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Veri setinin bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Özellik ölçeklendirilmesi yapılması
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train[:,3:] = st.fit_transform(x_train[:,3:])
x_test[:,3:] = st.transform(x_test[:,3:])