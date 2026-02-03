import json                     # per treballar amb fitxers json
import numpy as np              # estructures de dades
import matplotlib.pyplot as plt # per dibuixar
import scipy.io as sio          # entrada/sortida d'audio
from wav2vec import cutvowel, wav2vec  # el nostre mòdul


with open("./vowels/alex.json") as f:
    data = json.load(f)

print("Número de segments:", len(data))
print(data[30])  # comprovem l'estructura


# X -> vectors de característiques (formants)
# y -> etiqueta de la vocal

X = []
y = []

wav_file = "./vowels/alex.wav"

for i in range(len(data)):

    # llegim informació del segment
    start = float(data[i]["start"])
    end   = float(data[i]["end"])
    vocal = data[i]["vocal"]

    # retallem l'audio
    Fs, cut = cutvowel(wav_file, start, end)

    # descartem segments massa curts
    if len(cut) < 100:
        continue

    # extraiem els formants
    vec = wav2vec(cut, Fs)

    X.append(vec)
    y.append(vocal)

# passem a numpy
X = np.array(X)
y = np.array(y)

print("Dimensions X:", X.shape)
print("Dimensions y:", y.shape)


# VISUALITZACIÓ DELS FORMANTS

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


# NORMALITZACIÓ DE DADES

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xn = scaler.fit_transform(X)


# SEPARACIÓ TRAIN / TEST

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    Xn, y, test_size=0.2, random_state=0, stratify=y
)


# CLASSIFICADORS

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# definim tots els models
models = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=0)
}

# entrenem i avaluem tots els models
for name, model in models.items():
    
    # entrenament
    model.fit(X_train, y_train)
    
    # predicció
    y_pred = model.predict(X_test)
    
    # metrics
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n==================== {name} ====================")
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
