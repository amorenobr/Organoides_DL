import pandas as pd

from Organoides_DL.graficas import graficar_histograma

df = pd.read_csv("data/processed/area_safranina.csv")
fig_safranina = graficar_histograma(
    df["porcentaje_area"],
    title="PORCENTAJE DE AREA TEÑIDA CON SAFRANINA",
    xlabel="Area teñida (%)",
    ylabel="Numero de imagenes",
    bins=range(0, 101, 10),
    cortes=[10, 50],
)
fig_safranina.savefig(
    "reports/figuras/histograma_safranina.png", dpi="figure", bbox_inches="tight"
)
