#Kitaplıkları içe aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Veri setinin okunması
dataset = pd.read_csv("Salary_Data.csv")

#Özellik matrisi ve bağımlı değişken matrisinin belirlenmesi
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Veri setinin bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Basit Doğrusal Regresyon modelinin içe aktarılması
from sklearn.linear_model import LinearRegression

#Eğtim setini Basit Doğrusal Regresyon modeli üzerinde eğitmek
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Test seti sonuçlarını tahmin etme
y_pred = regressor.predict(x_test)

#Eğitim seti sonuçlarını görselleştirme
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Test seti sonuçlarını görselleştirme
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()