import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from minisom import MiniSom
import numpy as np


class Mapa:

    def __init__(self, som: MiniSom, X: np.ndarray, labels: list, size: int, fig_size=(14, 14)):
        self.som = som
        self.X = X
        self.labels = labels
        self.size = size
        self.fig_size = fig_size

        self.country_map = self.som.labels_map(self.X, self.labels)
        self.distance_map = self.som.distance_map().T

    def plot(self, labels_colors: list = None, legend_colors: dict = None):

        country_map = self.country_map
        fig_size = self.fig_size
        distance_map = self.distance_map
        size = self.size

        if labels_colors:
            # Label de cores por país
            colors_dict = dict(zip(self.labels, labels_colors))

        # colocando nome dos países e a respectiva cor de acordo com nível de democracia
        plt.figure(figsize=fig_size)
        for p, countries in country_map.items():
            labels = list(countries)
            x = p[0] + .1
            y = p[1] - .3
            for i, c in enumerate(countries):
                off_set = (i + 1) / len(countries) - 0.05
                if labels_colors:
                    plt.text(x, y + off_set, c, color=colors_dict[c], fontsize=10)
                else:
                    plt.text(x, y + off_set, c, color='black', fontsize=10)

        # Plotando a matriz de pesos
        plt.pcolor(distance_map, cmap='gray_r', alpha=.2)
        plt.xticks(np.arange(size + 1))
        plt.yticks(np.arange(size + 1))
        plt.grid()

        if labels_colors:
            # Criando as legendas
            legend_elements = [Patch(facecolor=color,
                                     edgecolor='w',
                                     label=label) for label, color in legend_colors.items()]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
        plt.show()


if __name__ == '__main__':
    pass

