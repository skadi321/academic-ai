import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATAPATH = r"C:\Users\Nikola\Downloads\archive\dataset.csv"
data = pd.read_csv(DATAPATH)
le = LabelEncoder()

y = le.fit_transform(data['Status'])
X = data.drop('Status', axis=1)


X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Assuming there are 3 classes
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

y_pred = np.argmax(model.predict(X_val), axis=-1)
f1 = f1_score(y_val, y_pred, average='weighted')
print("F1 Score:", f1)


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Odustao", "Upisan", "Diplomirao"], yticklabels=["Odustao", "Upisan", "Diplomirao"])
plt.xlabel("Predikcije Statusa")
plt.ylabel("Stvarni status")
plt.title("Matrica zabune")


save_path = r"C:\Users\Nikola\Desktop\NeuralNetwork"  # Change this to your desired directory
plot_filename = "confusion_matrix.png"
plot_path = f"{save_path}\\{plot_filename}"
plt.savefig(plot_path)
plt.show()
