#Kitaplıkları içe aktarma
import numpy as np
import pandas as pd
import tensorflow as tf

#Veri kümesini içe aktarma
dataset = pd.read_csv('Churn_Modelling.csv')

#Özellik matrisi ve bağımlı değişken matrisinin belirlenmesi
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Toplam satır sütun sayısı
print(f'Number of rows:', dataset.shape[0])
print(f'Number of columns:', dataset.shape[1])

#Veri setinin özellikleri
dataset.info()

#Etiket "Cinsiyet" sütununu kodluyor
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#"Coğrafya" sütununu sıcak bir şekilde kodlama
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Veri kümesini Eğitim kümesi ve Test kümesine bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Özellik ölçeklendirilmesinin uygulanması
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#YSA'nın başlatılması
ann = tf.keras.models.Sequential()

#Giriş katmanını ve ilk gizli katmanı ekleme
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#İkinci gizli katmanın eklenmesi
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Çıkış katmanını ekleme
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#YSA'nın derlenmesi
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#YSA'nın Eğitim Seti Üzerinde Eğitimi
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Tek bir gözlemin sonucunu tahmin etmek
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#Test seti sonuçlarını tahmin etme
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Karışıklık Matrisinin oluşturulması
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#Karışıklık Matrisinin çizdirilmesi
import matplotlib.pyplot as plt
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [True, False])

cm_display.plot()
plt.show()

#Performans metriklerinin tablosu
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

#ROC eğrisinin çizdirilmesi
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2, color= 'teal')
plt.plot([0,1], [0,1], 'r--' )
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()