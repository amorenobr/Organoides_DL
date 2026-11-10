import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

huellas = pd.read_csv("data/processed/huellas_gris.csv")
huellas = huellas[huellas["grupo"] != "figuras"]
huellas["paciente"] = huellas["imagen"].str.split("_").str[0]

pacientes = (
    huellas.drop(columns="imagen").groupby(["paciente", "grupo"]).mean().reset_index()
)
y = (pacientes["grupo"] == "sano").astype(int)
X = pacientes.drop(columns=["paciente", "grupo"]).values

modelo = LogisticRegression(max_iter=5000)
prediccion = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
entrenamiento = modelo.fit(X, y).predict(X)

print(f"Dejar un paciente fuera: {(prediccion == y).sum()}/{len(y)}")
print(f"Entrenamiento: {(entrenamiento == y).sum()}/{len(y)}")
