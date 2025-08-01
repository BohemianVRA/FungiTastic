import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import cv2

# Fallback color palette for unknown labels
FALLBACK_COLORS = plt.cm.Set3(np.linspace(0, 1, 20))  # 20 distinct colors

import os

import numpy as np
import matplotlib
from matplotlib import style, colormaps as mpl_cm
import matplotlib.pyplot as plt

from dataset.fungi import FungiTastic

# Default color maps
DEFAULT_CMAP_SEQ = 'cividis_r'
DEFAULT_CMAP_QUAL = 'Accent'


class FungiTasticVis(FungiTastic):
    """
    Aggregates utilities for visualization purposes.
    Inherits from FungiTastic and provides methods to visualize data such as species,
    genus, habitat, and substrate frequencies, as well as example images.
    """

    habitat2short = {
    'Acidic oak woodland': 'acidic oak',
    'Deciduous woodland': 'deciduous woodland',
    'Unmanaged deciduous woodland': 'unmanaged deciduous',
    'coniferous woodland/plantation': 'coniferous woodland',
    'Mixed woodland (with coniferous and deciduous trees)': 'mixed woodland',
    'roadside': 'roadside',
    'heath': 'heath',
    'natural grassland': 'natural grassland',
    'park/churchyard': 'park/churchyard',
    'Thorny scrubland': 'thorny scrubland',
    'ditch': 'ditch',
    'Unmanaged coniferous woodland': 'unmanaged coniferous',
    'bog': 'bog',
    'wooded meadow, grazing forest': 'wooded meadow',
    'salt meadow': 'salt meadow',
    'dune': 'dune',
    'Willow scrubland': 'willow scrubland',
    'hedgerow': 'hedgerow',
    'Forest bog': 'forest bog',
    'lawn': 'lawn',
    'improved grassland': 'improved grassland',
    'garden': 'garden',
    'Bog woodland': 'bog woodland',
    'other habitat': 'other habitat',
    'gravel or clay pit': 'gravel/clay',
    'meadow': 'meadow',
    'fallow field': 'fallow field',
    np.nan: 'none'
    }

    substrate2short = {
        'soil': 'soil',
        'leaf or needle litter': 'leaf/needle litter',
        'stems of herbs, grass etc': 'stems',
        'dead wood (including bark)': 'dead wood',
        'bark of living trees': 'bark living',
        'wood chips or mulch': 'wood chips/mulch',
        'mosses': 'mosses',
        'wood and roots of living trees': 'wood living',
        'peat mosses': 'peat mosses',
        'faeces': 'faeces',
        'dead stems of herbs, grass etc': 'dead stems',
        'other substrate': 'other',
        'living stems of herbs, grass etc': 'living stems',
        'cones': 'cones',
        'fruits': 'fruits',
        'fungi': 'fungi',
        np.nan: 'none'
    }

    @property
    def name(self):
        return f'Fungitastic-{self.data_subset}-{self.split}-{self.task}'

    def plot_label_freq(self, save_path=None, save_name='label_freq.pdf', show_names=True):
        return self.plot_species_freq(save_path=save_path, save_name=save_name, show_names=show_names)

    def plot_species_freq(self, save_path=None, save_name='species_freq.pdf', figsize=(30, 4), show_names=True, base_fs=15,
                          split_by_genus=False, ax=None, show=True, title=None, ret_species2color=False, species2color=None,
                          category_id_order=None, ret_category_id_order=False, y_labels_off=False, y_max=None, ret_y_max=False):

        """
        Plot the frequency of species in the dataset.
        Parameters:
            - save_path: Optional path to save the plot.
            - save_name: Name of the file to save the plot as.
            - figsize: Size of the figure.
            - show_names: Whether to show species names on the x-axis.
            - base_fs: Base font size for the plot.
            - split_by_genus: Whether to color bars by genus.
            - ax: Axis to plot on.
            - show: Whether to display the plot.
            - title: Title of the plot.
            - ret_species2color: Whether to return the species-to-color mapping.
            - species2color: Optional mapping of species to colors.
            - category_id_order: Order of categories to plot.
            - ret_category_id_order: Whether to return the category order.
            - y_labels_off: Whether to hide y-axis labels.
            - y_max: Maximum y-axis value.
            - ret_y_max: Whether to return the y-axis maximum value.
        """

        matplotlib.rc('axes', edgecolor='black')
        if title is None:
            f"{self.name} label frequency"
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()
        value_counts = self.df.species.value_counts()
        specie2genus = {k: v[0] for k, v in self.df.groupby('species')['genus'].unique().to_dict().items()}

        if category_id_order is not None:
            # only keep those that are in the split
            category_id_order = [c for c in category_id_order if c in self.df.category_id.unique()]
            sorted_classes = [self.category_id2label[c] for c in category_id_order]
            value_counts = value_counts[sorted_classes]
        else:
            category_id_order = [self.label2category_id[l] for l in value_counts.index]

        if split_by_genus:
            #  each genus has a different bar color
            genus2color = {genus: mpl_cm.get_cmap(DEFAULT_CMAP_QUAL)(i / len(self.df.genus.unique())) for i, genus in enumerate(self.df.genus.unique())}
            colors = [genus2color[specie2genus[specie]] for specie in value_counts.index]
        else:
            if species2color is None:
                colors = mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(value_counts.values / max(value_counts.values))
                species2color = {specie: color for specie, color in zip(value_counts.index, colors)}
            else:
                colors = [species2color[specie] for specie in value_counts.index]


        value_counts.plot(
            kind='bar',
            width=0.8,
            # colorcode the bars by value from dark to light
            color=colors,
            legend=False,
            ax=ax
        )
        if not show_names:
            ax.set_xticks([])
        else:
            ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=90, fontsize=0.4 * base_fs)
        # place the title in the middle top, inside the plot, set background to 0.9 transparent white
        ax.set_title(title, fontsize=base_fs, y=0.83, backgroundcolor=(1, 1, 1, 0.9))
        # increase yticks size
        if not y_labels_off:
            ax.set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels(), fontsize=base_fs * 0.6)
        else:
            # set empty labels, ticks from (0 to max value, 5 ticks)
            ticks = np.linspace(0, max(value_counts.values) if y_max is None else y_max, 5)
            ax.set_yticks(labels=['' for _ in ticks], ticks=ticks)
            #     also set ylim to the max
            if y_max is None:
                ax.set_ylim(0, max(value_counts.values))
                y_max = max(value_counts.values)
                labels = ['' for _ in ticks]
                labels[0] = 0
                labels[-1] = y_max
                ax.set_yticks(labels=labels, ticks=ticks, fontsize=base_fs * 0.7)
            else:
                ax.set_ylim(0, y_max)
        ax.set_xlabel("")
        if split_by_genus:
            # show legend
            for genus, color in genus2color.items():
                plt.plot([], [], color=color, label=genus)
            # remove the last item form the legend ("count"), which was set by pandas in df.plot()
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend(handles[:-1], labels[:-1], title='Genus', title_fontsize=base_fs, fontsize=base_fs * 0.8)
        # only show horizontal grid
        ax.grid(axis='x')
        # make the grid almsot transparent
        ax.grid(alpha=0.2)

        if save_path:
            save_path = os.path.join(save_path, save_name)
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

        ret = {}
        if ret_species2color:
            ret['species2color'] = species2color
        if ret_category_id_order:
            ret['category_id_order'] = category_id_order
        if ret_y_max:
            ret['y_max'] = y_max
        return ret

    def plot_genus_freq(self, save_path=None, save_name='genus_freq.pdf', figsize=(10, 4)):
        """Plot the frequency of genus in the dataset."""
        plt.figure(figsize=figsize)
        self.df.genus.value_counts().plot(
            kind='bar',
            # increase bar width
            width=0.8,
            # colorcode the bars by value from dark to light
            color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(self.df.genus.value_counts().values / max(self.df.genus.value_counts().values))
        )
        plt.xticks(rotation=90)
        plt.title(f"{self.name} genus frequency", fontsize=14)
        plt.xlabel("")
        plt.xticks(fontsize=14)
        # only show horizontal grid
        plt.grid(axis='x')

        if save_path:
            save_path = os.path.join(save_path, save_name)
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_habitat_freq(self, figsize=(10, 4)):
        """Plot the frequency of habitats in the dataset."""
        plt.figure(figsize=figsize)
        # replace habitat names with shorter versions
        habitat = self.df['habitat'].apply(lambda x: self.habitat2short[x] if x in self.habitat2short else x)
        habitat.value_counts().plot(
            kind='bar',
            width=0.8,
            color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(habitat.value_counts().values / max(habitat.value_counts().values)),
        )
        # set the labels
        plt.xticks(range(len(habitat.value_counts().index)), habitat.value_counts().index, rotation=90, fontsize=15)
        plt.title(f"{self.name} habitat frequency", fontsize=14)
        plt.xlabel("")        # only show horizontal grid
        plt.grid(axis='x')
        plt.show()

    def plot_substrat_freq(self, figsize=(10, 4)):
        """Plot the frequency of substrates in the dataset."""
        plt.figure(figsize=figsize)
        # replace substrat names with shorter versions
        substrat = self.df['substrate'].apply(lambda x: self.substrate2short[x] if x in self.substrate2short else x)
        substrat.value_counts().plot(
            kind='bar',
            width=0.8,
            color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(substrat.value_counts().values / max(substrat.value_counts().values)),
        )
        # set the labels
        plt.xticks(range(len(substrat.value_counts().index)), substrat.value_counts().index, rotation=90, fontsize=15)
        plt.title(f"{self.name} substrat frequency", fontsize=14)
        plt.xlabel("")        # only show horizontal grid
        plt.grid(axis='x')
        plt.show()

    def show_substrate_examples(self, sample_n=3, substrate_n=5):
        """
        Show sample_n examples of the substrate_n most common substrates.
        :param sample_n:
        :param substrate_n:
        :return:
        """
        # keep only first sample from one observation id
        df = self.df.drop_duplicates(subset='observationID')
        substrates = df.substrate.value_counts().index[:substrate_n]
        n_rows, n_cols = sample_n, substrate_n
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        for i, substrate in enumerate(substrates):
            idxs = self.df[self.df.substrate == substrate].sample(sample_n).index
            for j, idx in enumerate(idxs):
                image, category_id, file_path = self.__getitem__(idx)
                # resize image to 256x256
                image = image.resize((256, 256))
                axs[j, i].imshow(image)
                axs[j, i].axis('off')
                if j == 0:
                    multiline_title = self.substrate2short[substrate].replace(' ', '\n')
                    axs[j, i].set_title(f"{multiline_title}", fontsize=12)
        plt.tight_layout()
        plt.suptitle(f"{self.name} substrates", fontsize=15, y=1.05)
        plt.show()

    def show_habitat_examples(self, sample_n=3, habitat_n=5):
        """
        Show sample_n examples of the habitat_n most common habitats.
        :param sample_n:
        :param habitat_n:
        :return:
        """
        # keep only first sample from one observation id
        df = self.df.drop_duplicates(subset='observationID')
        habitats = df.habitat.value_counts().index[:habitat_n]
        n_rows, n_cols = sample_n, habitat_n
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        for i, habitat in enumerate(habitats):
            idxs = self.df[self.df.habitat == habitat].sample(sample_n).index
            for j, idx in enumerate(idxs):
                image, category_id, file_path = self.__getitem__(idx)
                # resize image to 256x256
                image = image.resize((256, 256))
                axs[j, i].imshow(image)
                axs[j, i].axis('off')
                if j == 0:
                    # make the title multiline if it has multiple words
                    multiline_title = self.habitat2short[habitat].replace(' ', '\n')
                    axs[j, i].set_title(f"{multiline_title}", fontsize=12)
        plt.tight_layout()
        # title below the subplots
        plt.suptitle(f"{self.name} habitats", fontsize=15, y=1.05)
        plt.show()

    def show_all_class_examples(self, sample_n=3, class_n=5, category_idxs=None):
        """
        Show sample_n examples of category_idxs classes, if provided. Othervise, class_n
         most frequent classes are shown.
        :param sample_n:
        :param class_n:
        :param category_idxs:
        :return:
        """

        if category_idxs is None:
            category_idxs = self.df.category_id.value_counts().index[:class_n]
        n_rows, n_cols = sample_n, class_n
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        for i, category_id in enumerate(category_idxs):
            idxs = self.get_category_idxs(category_id)
            for j, idx in enumerate(idxs[:sample_n]):
                image, category_id, file_path = self.__getitem__(idx)
                # resize image to 256x256
                image = image.resize((256, 256))
                axs[j, i].imshow(image)
                axs[j, i].axis('off')
                if j == 0:
                    multiline_title = self.category_id2label[category_id].replace(' ', '\n')
                    axs[j, i].set_title(f"{multiline_title}", fontsize=12)
        plt.tight_layout()
        plt.suptitle(f"{self.name} classes", fontsize=15, y=1.05)
        plt.show()

