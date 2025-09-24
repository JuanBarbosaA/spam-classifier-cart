import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

os.makedirs("images", exist_ok=True)

data = pd.read_csv("data/dataset_spam.csv")

features = [
    "email_length", "word_count", "num_links", "num_exclamations",
    "num_dollars", "num_upper_words", "num_spam_words",
    "has_numbers", "subject_length", "num_lines"
]
X = data[features]
y = data["label"]

results = {"accuracy": [], "f1": []}

# -------------------- ENTRENAMIENTO REPETIDO --------------------
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=None
    )

    model = DecisionTreeClassifier(criterion="gini", random_state=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results["accuracy"].append(acc)
    results["f1"].append(f1)

# -------------------- CÁLCULO DE MÉTRICAS --------------------
z_acc = stats.zscore(results["accuracy"])
z_f1 = stats.zscore(results["f1"])

df_results = pd.DataFrame({
    "accuracy": results["accuracy"],
    "f1": results["f1"],
    "zscore_accuracy": z_acc,
    "zscore_f1": z_f1
})

summary = df_results.describe()
print("Resumen estadístico de las métricas:\n")
print(summary)

print("\nPromedios y desviaciones:")
print("Accuracy promedio:", np.mean(results["accuracy"]))
print("F1 Score promedio:", np.mean(results["f1"]))
print("Desviación Accuracy:", np.std(results["accuracy"]))
print("Desviación F1:", np.std(results["f1"]))

outliers_acc = df_results[df_results["zscore_accuracy"].abs() > 2]
outliers_f1 = df_results[df_results["zscore_f1"].abs() > 2]

print("\nEjecuciones con Accuracy atípico:\n", outliers_acc)
print("\nEjecuciones con F1 Score atípico:\n", outliers_f1)

# -------------------- GRAFICAR RESULTADOS --------------------
plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.plot(df_results["accuracy"], marker="o")
plt.title("Accuracy en 50 ejecuciones")
plt.xlabel("Ejecución")
plt.ylabel("Accuracy")

plt.subplot(1,3,2)
plt.plot(df_results["f1"], marker="o", color="orange")
plt.title("F1 Score en 50 ejecuciones")
plt.xlabel("Ejecución")
plt.ylabel("F1 Score")

plt.subplot(1,3,3)
plt.plot(df_results["zscore_accuracy"], marker="o", color="green", label="Z Accuracy")
plt.plot(df_results["zscore_f1"], marker="x", color="red", label="Z F1")
plt.title("Z Scores")
plt.xlabel("Ejecución")
plt.ylabel("Z Score")
plt.legend()

plt.tight_layout()
plt.savefig("images/resultados_metricas.png")  
plt.show()

# -------------------- HISTOGRAMAS --------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df_results["accuracy"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribución de Accuracy")

plt.subplot(1,2,2)
plt.hist(df_results["f1"], bins=10, color="orange", edgecolor="black")
plt.title("Distribución de F1 Score")

plt.tight_layout()
plt.savefig("images/histogramas_metricas.png") 
plt.show()

# -------------------- GUARDAR RESULTADOS --------------------
df_results.to_csv("results/resultados_spam.csv", index=False)
print("\nResultados guardados en 'resultados_spam.csv'")
print("Gráficas guardadas en la carpeta 'images/'")
