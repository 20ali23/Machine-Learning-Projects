#Kitaplıkları içe aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Veri setinin okunması
dataset = pd.read_csv("Position_Salaries.csv")

#Özellik matrisi ve bağımlı değişken matrisinin belirlenmesi
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Karar Ağacı Regresyon modelini tüm veri kümesi üzerinde eğitme
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#Yeni bir sonuç tahmin etmek
regressor.predict([[6.5]])

#Karar Ağacı Regresyon sonuçlarının görselleştirilmesi (daha yüksek çözünürlük)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

