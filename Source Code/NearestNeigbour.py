import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier



DATAPATH = r"C:\Users\Nikola\Downloads\archive\dataset.csv"

data = pd.read_csv(DATAPATH)

le = LabelEncoder()


y = le.fit_transform(data['Status'])
X = data.drop('Status', axis=1) 

X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25)



X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)



knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f1, end=", ")


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Odustao", "Upisan", "Diplomirao"], yticklabels=["Odustao", "Upisan", "Diplomirao"])
plt.xlabel("Predikcije Statusa")
plt.ylabel("Stvarni status")
plt.title("Matrica zabune")
plt.show()
