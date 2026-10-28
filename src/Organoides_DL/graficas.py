import matplotlib.pyplot as plt


def graficar_histograma(data, title, xlabel, ylabel, bins=10, cortes=None):

    fig = plt.figure(figsize=(15, 8))
    _, _, barras = plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)
    plt.bar_label(barras)

    if cortes is not None:
        for corte in cortes:
            plt.axvline(x=corte, color="red", linestyle="--", label=f"Corte: {corte}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(0, 101, 10))

    if cortes is not None:
        plt.legend()

    plt.grid(axis="y", alpha=0.75)

    return fig
