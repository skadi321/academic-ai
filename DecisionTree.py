import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score




DATAPATH = r"C:\Users\Nikola\Downloads\archive\dataset.csv"

data = pd.read_csv(DATAPATH)

le = LabelEncoder()


y = le.fit_transform(data['Status'])
X = data.drop('Status', axis=1) 

X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25)


y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)



clf = DecisionTreeClassifier(max_depth=6)    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
f1 = f1_score(y_val, y_pred, average='weighted') 
print("F1 Score:", f1)

    
cm = confusion_matrix(y_val, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Odustao", "Upisan", "Diplomirao"], yticklabels=["Odustao", "Upisan", "Diplomirao"])
plt.xlabel("Predikcije Statusa")
plt.ylabel("Stvarni status")
plt.title("Matrica zabune")
plt.show()
