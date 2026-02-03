import json
import numpy as np
import matplotlib.pyplot as plt
import librosa                  # per MFCC
from wav2vec import cutvowel, wav2vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# Carreguem JSON
# --------------------------
with open("./vowels/alex.json") as f:
    data = json.load(f)

print("Número de segments:", len(data))
print(data[30])

# --------------------------
# Extracció de característiques
# --------------------------
X = []
y = []

wav_file = "./vowels/alex.wav"

for i in range(len(data)):
    start = float(data[i]["start"])
    end   = float(data[i]["end"])
    vocal = data[i]["vocal"]

    Fs, cut = cutvowel(wav_file, start, end)
    
    if len(cut) < 100:
        continue

    # MFCCs + wav2vec combinats
    vec_wav2vec = wav2vec(cut, Fs)

    # MFCC amb librosa
    cut_float = cut.astype(float)
    mfccs = librosa.feature.mfcc(y=cut_float, sr=Fs, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)

    # combinem característiques
    vec = np.concatenate([vec_wav2vec, mfcc_mean])
    X.append(vec)
    y.append(vocal)

X = np.array(X)
y = np.array(y)

print("Dimensions X:", X.shape)
print("Dimensions y:", y.shape)

# --------------------------
# Visualització només F1/F2 de wav2vec
# --------------------------
plt.figure()
for v in np.unique(y):
    idx = y == v
    plt.scatter(X[idx,0], X[idx,1], label=v)

plt.xlabel("F1 (Hz)")
plt.ylabel("F2 (Hz)")
plt.title("Formants de les vocals")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# Normalització
# --------------------------
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# --------------------------
# Train/test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    Xn, y, test_size=0.2, random_state=0, stratify=y
)

# --------------------------
# CLASSIFICADORS
# --------------------------
models = {}

models["SVM"] = SVC(
    kernel="rbf",      # tipo de kernel
    C=10,              # parámetro de regularización
    gamma="scale",     # control del ancho del kernel
    class_weight="balanced"  # para clases desbalanceadas
)

# Altres models
models["KNN"] = KNeighborsClassifier(n_neighbors=5)
models["Logistic Regression"] = LogisticRegression(max_iter=500)
models["Decision Tree"] = DecisionTreeClassifier(max_depth=10, random_state=0)

# --------------------------
# Entrenament i avaluació
# --------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n==================== {name} ====================")
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
