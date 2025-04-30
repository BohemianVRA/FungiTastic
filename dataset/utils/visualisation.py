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
        """
        Generates a descriptive name for the current dataset and task configuration.

        Returns:
            str: The generated name string.
        """
        return f'Fungitastic-{self.data_subset}-{self.split}-{self.task}'

    def plot_label_freq(self, save_path=None, save_name='label_freq.pdf', show_names=True):
        """
        Plots the frequency of labels in the dataset.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            save_name (str, optional): Name of the file to save the plot as. Defaults to 'label_freq.pdf'.
            show_names (bool, optional): Whether to show label names on the x-axis. Defaults to True.

        Returns:
            None
        """
        return self.plot_species_freq(save_path=save_path, save_name=save_name,
                                      show_names=show_names)

    def plot_species_freq(self, save_path=None, save_name='species_freq.pdf', figsize=(30, 4),
                          show_names=True, base_fs=15,
                          split_by_genus=False, ax=None, show=True, title=None,
                          ret_species2color=False, species2color=None,
                          category_id_order=None, ret_category_id_order=False, y_labels_off=False,
                          y_max=None, ret_y_max=False):
        """
        Plots the frequency of species in the dataset.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            save_name (str, optional): Name of the file to save the plot as. Defaults to 'species_freq.pdf'.
            figsize (tuple[int, int], optional): Size of the figure in inches. Defaults to (30, 4).
            show_names (bool, optional): Whether to show species names on the x-axis. Defaults to True.
            base_fs (int, optional): Base font size for the plot. Defaults to 15.
            split_by_genus (bool, optional): Whether to color bars by genus. Defaults to False.
            ax (matplotlib.axes.Axes, optional): Axis to plot on. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            title (str, optional): Title of the plot. Defaults to None.
            ret_species2color (bool, optional): Whether to return the species-to-color mapping. Defaults to False.
            species2color (dict, optional): Mapping of species to colors. Defaults to None.
            category_id_order (list[int], optional): Order of categories to plot. Defaults to None.
            ret_category_id_order (bool, optional): Whether to return the category order. Defaults to False.
            y_labels_off (bool, optional): Whether to hide y-axis labels. Defaults to False.
            y_max (int, optional): Maximum y-axis value. Defaults to None.
            ret_y_max (bool, optional): Whether to return the y-axis maximum value. Defaults to False.

        Returns:
            dict: Contains optional return values based on provided arguments:
                  'species2color', 'category_id_order', 'y_max'.
        """
        matplotlib.rc('axes', edgecolor='black')

        if title is None:
            title = f"{self.name} label frequency"

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        value_counts = self.df.species.value_counts()
        specie2genus = {k: v[0] for k, v in
                        self.df.groupby('species')['genus'].unique().to_dict().items()}

        if category_id_order is not None:
            category_id_order = [c for c in category_id_order if c in self.df.category_id.unique()]
            sorted_classes = [self.category_id2label[c] for c in category_id_order]
            value_counts = value_counts[sorted_classes]
        else:
            category_id_order = [self.label2category_id[l] for l in value_counts.index]

        if split_by_genus:
            genus2color = {
                genus: mpl_cm.get_cmap(DEFAULT_CMAP_QUAL)(i / len(self.df.genus.unique())) for
                i, genus in enumerate(self.df.genus.unique())}
            colors = [genus2color[specie2genus[specie]] for specie in value_counts.index]
        else:
            if species2color is None:
                colors = mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(
                    value_counts.values / max(value_counts.values))
                species2color = {specie: color for specie, color in zip(value_counts.index, colors)}
            else:
                colors = [species2color[specie] for specie in value_counts.index]

        value_counts.plot(
            kind='bar',
            width=0.8,
            color=colors,
            legend=False,
            ax=ax
        )

        if not show_names:
            ax.set_xticks([])
        else:
            ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), rotation=90,
                          fontsize=0.4 * base_fs)

        ax.set_title(title, fontsize=base_fs, y=0.83, backgroundcolor=(1, 1, 1, 0.9))

        if not y_labels_off:
            ax.set_yticks(ticks=ax.get_yticks(), labels=ax.get_yticklabels(),
                          fontsize=base_fs * 0.6)
        else:
            ticks = np.linspace(0, max(value_counts.values) if y_max is None else y_max, 5)
            ax.set_yticks(labels=['' for _ in ticks], ticks=ticks)
            if y_max is None:
                y_max = max(value_counts.values)
            ax.set_ylim(0, y_max)
            labels = ['' for _ in ticks]
            labels[0] = 0
            labels[-1] = y_max
            ax.set_yticks(labels=labels, ticks=ticks, fontsize=base_fs * 0.7)

        ax.set_xlabel("")

        if split_by_genus:
            for genus, color in genus2color.items():
                plt.plot([], [], color=color, label=genus)
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend(handles[:-1], labels[:-1], title='Genus', title_fontsize=base_fs,
                      fontsize=base_fs * 0.8)

        ax.grid(axis='x', alpha=0.2)

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
        """
        Plots the frequency of genus in the dataset.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to None.
            save_name (str, optional): Name of the file to save the plot as. Defaults to 'genus_freq.pdf'.
            figsize (tuple[int, int], optional): Size of the figure in inches. Defaults to (10, 4).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        genus_counts = self.df.genus.value_counts()
        genus_counts.plot(
            kind='bar',
            width=0.8,
            color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(genus_counts.values / max(genus_counts.values))
        )
        plt.xticks(rotation=90, fontsize=14)
        plt.title(f"{self.name} genus frequency", fontsize=14)
        plt.xlabel("")
        plt.grid(axis='x', alpha=0.2)

        if save_path:
            save_path = os.path.join(save_path, save_name)
            plt.savefig(save_path, dpi=300)

        plt.show()

    def plot_habitat_freq(self, figsize=(10, 4)):
        """
        Plots the frequency of habitats in the dataset.

        Args:
            figsize (tuple[int, int], optional): The size of the figure in inches.
                Defaults to (10, 4).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        habitat_counts = self.df.habitat.map(self.habitat2short).value_counts()
        habitat_counts.plot(kind='bar', width=0.8, color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(
            habitat_counts.values / max(habitat_counts.values)))
        plt.xticks(rotation=90, fontsize=14)
        plt.title(f"{self.name} habitat frequency", fontsize=14)
        plt.xlabel("")
        plt.grid(axis='x', alpha=0.2)
        plt.show()

    def plot_substrate_freq(self, figsize=(10, 4)):
        """
        Plots the frequency of substrates in the dataset.

        Args:
            figsize (tuple[int, int], optional): The size of the figure in inches.
                Defaults to (10, 4).

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        substrate_counts = self.df.substrate.map(self.substrate2short).value_counts()
        substrate_counts.plot(kind='bar', width=0.8, color=mpl_cm.get_cmap(DEFAULT_CMAP_SEQ)(
            substrate_counts.values / max(substrate_counts.values)))
        plt.xticks(rotation=90, fontsize=14)
        plt.title(f"{self.name} substrate frequency", fontsize=14)
        plt.xlabel("")
        plt.grid(axis='x', alpha=0.2)
        plt.show()

    def plot_example_images(self, n=5, save_path=None, save_name='example_images.pdf'):
        """
        Displays example images from the dataset.

        Args:
            n (int, optional): Number of images to display. Defaults to 5.
            save_path (str, optional): Path to save the images. Defaults to None.
            save_name (str, optional): Name of the file to save the images as. Defaults to 'example_images.pdf'.

        Returns:
            None
        """
        images = self.df.image_path.sample(n).values
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))

        for img, ax in zip(images, axes):
            image = plt.imread(img)
            ax.imshow(image)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            save_path = os.path.join(save_path, save_name)
            plt.savefig(save_path, dpi=300)

        plt.show()
