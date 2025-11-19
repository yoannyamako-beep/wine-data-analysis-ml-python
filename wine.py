

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from numpy.fft import fft


wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = wine.target

print("Nombre de lignes et de colonnes :", X_wine.shape)
print("Colonnes Wine   :" list(X_wine.columns))
print()



corr = X_wine.corr()

corr_no_diag = corr.where(~np.eye(corr.shape[0], dtype=bool))
max_pair = None
max_val = 0

for i in corr.index:
    for j in corr.columns:
        val = corr.loc[i, j]
        if pd.notna(val) and abs(val) > abs(max_val):
            max_val = val
            max_pair = (i, j)

var1, var2 = max_pair
print("Paire la plus corrélée : ", var1, "-", var2)
print("Coefficient de corrélation de Pearson :", max_val)
print()

plt.figure(figsize=(6, 4))
plt.scatter(X_wine[var1], X_wine[var2], alpha=0.7)
plt.xlabel(var1)
plt.ylabel(var2)
plt.title(f"Relation entre {var1} et {var2}\nCorrélation = {max_val:.3f}")
plt.tight_layout()
plt.show()

signal = X_wine[var1].values
fft_vals = np.abs(fft(signal))
freqs = np.fft.fftfreq(len(signal))


mask_pos = freqs >= 0
plt.figure(figsize=(6, 4))
plt.plot(freqs[mask_pos], fft_vals[mask_pos])
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.title(f"Transformée de Fourier de la variable {var1}")
plt.tight_layout()
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
)

clf_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
clf_knn.fit(X_train, y_train)

print("Accuracy KNN (Wine, split 70/30) :", clf_knn.score(X_test, y_test))
print()

test = [11, 1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1, 111]
new_d = [13, 2, 2, 20, 99, 2, 2, 0.4, 2, 5, 1, 2.5, 500]

pred_test = clf_knn.predict([test])[0]
pred_new_d = clf_knn.predict([new_d])[0]

print("Classe prédite pour 'test'  :", pred_test)
print("Classe prédite pour 'new_d' :", pred_new_d)
print()




