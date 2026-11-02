
import pandas as pd
from scipy.stats import kruskal, spearmanr

clases = pd.read_csv("data/processed/safranina_folds.csv")[["ID_imagen", "clase"]]
area = pd.read_csv("data/processed/area_safranina.csv")
info = clases.merge(area, on="ID_imagen").rename(columns={"ID_imagen": "imagen"})


def calcular_eje(pacientes):
    sano = pacientes.loc["sano"].mean()
    degenerado = pacientes.loc["degenerado"].mean()
    return degenerado, sano - degenerado


for archivo in ["huellas", "huellas_gris"]:
    d = pd.read_csv(f"data/processed/{archivo}.csv").copy()
    numeros = d.columns[2:]
    d["paciente"] = d["imagen"].str.split("_").str[0]
    pacientes = d[d["grupo"] != "figuras"].groupby(["grupo", "paciente"])[numeros].mean()
    print(f"--- {archivo} ---")

    # 1
    aciertos = 0
    for (grupo, paciente), huella in pacientes.iterrows():
        degenerado, eje = calcular_eje(pacientes.drop(index=(grupo, paciente)))
        indice = (huella - degenerado) @ eje / (eje @ eje)
        aciertos += (indice > 0.5) == (grupo == "sano")
    print(f"Validación: {aciertos} de {len(pacientes)} pacientes bien ubicados")

    # 2
    degenerado, eje = calcular_eje(pacientes)
    organoides = d[d["grupo"] == "figuras"].merge(info, on="imagen")
    organoides["indice"] = (organoides[numeros] - degenerado) @ eje / (eje @ eje)
    print(organoides.groupby("clase")["indice"].mean().round(2))

    # 3
    rho, p = spearmanr(organoides["porcentaje_area"], organoides["indice"])
    grupos = [g["indice"] for _, g in organoides.groupby("clase")]
    print(f"Spearman con % de área: rho = {rho:.2f}, p = {p:.4f}")
    print(f"Kruskal-Wallis entre clases: p = {kruskal(*grupos).pvalue:.4f}")

    organoides[["imagen", "clase", "indice"]].to_csv(f"data/processed/indice_{archivo}.csv", index=False)