#Kitaplıkları içe aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Veri setinin okunması
dataset = pd.read_csv("Position_Salaries.csv")

#Özellik matrisi ve bağımlı değişken matrisinin belirlenmesi
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Doğrusal Regresyon modelini tüm veri kümesi üzerinde eğitme
from sklearn.linear_model import LinearRegression

#Eğtim setini Basit Doğrusal Regresyon modeli üzerinde eğitmek
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Polinom Regresyon modelini tüm veri kümesi üzerinde eğitme
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=10)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Doğrusal Regresyon sonuçlarının görselleştirilmesi
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Polinom Regresyon sonuçlarının görselleştirilmesi
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Polinom Regresyon sonuçlarının görselleştirilmesi (daha yüksek çözünürlük ve daha düzgün eğri için)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Doğrusal Regresyon ile yeni bir sonucu tahmin etme
lin_reg.predict([[6.5]])

#Polinom Regresyon ile yeni bir sonuç tahmin etme
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))