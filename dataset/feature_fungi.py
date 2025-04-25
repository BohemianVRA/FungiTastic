import os
from typing import Tuple, Any
import yaml


import torch
import torchvision.transforms as T

import pandas as pd
from dataset.fungi import FungiTastic


class FeatureFungiTastic(FungiTastic):
    def __init__(self, root: str, features_file: str, data_subset: str = 'Mini', split: str = 'val', size: str = '300',
                 task: str = 'closed', transform: T.Compose = None, rescale=True, **kwargs):
        super().__init__(
            root=root,
            data_subset=data_subset,
            split=split,
            size=size,
            task=task,
            transform=transform,
            **kwargs
        )
        embeddings = pd.read_hdf(features_file)

        if rescale:
            embeddings['embedding'] = embeddings['embedding'].apply(self.rescale_embedding)

        self.embeddings = embeddings
        self.emb_dim = self.embeddings['embedding'].iloc[0].shape[1]

    def check_integrity(self):
        #     make sure the embedding im_name is the same as df filename for all samples
        emb_names = self.embeddings['im_name'].values
        df_names = self.df['filename'].values
        assert (emb_names == df_names).all()
        print('Integrity check passed!')

    @staticmethod
    def rescale_embedding(embedding):
        # rescale the embedding from np.uint8 back to the range [-1, 1]
        return embedding / 255.0 * 2 - 1

    def get_embeddings_for_class(self, id):
        # return the embeddings for class class_idx
        class_idxs = self.df[self.df['category_id'] == id].index
        return self.embeddings.iloc[class_idxs]['embedding']

    def __getitem__(self, index: int, ret_image=False) -> Tuple[Any, Any, Any]:
        image, class_id, file_path = super().__getitem__(index)
        emb = torch.tensor(self.embeddings.iloc[index]['embedding'], dtype=torch.float32).squeeze()

        if ret_image:
            return image, emb, class_id, file_path
        else:
            return emb, class_id, file_path

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Use LaTeX-like font for paper visualization if needed and LaTeX is installed
    if False:  # Change to True if you want LaTeX-like font
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    split = 'val'
    with open('../config/path.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    model = 'dinov2'
    features_file = os.path.join(cfg.feature_path, model, f"224x224_no_micro_{split}.h5")

    valset = FeatureFungiTastic(
        root=cfg.data_path,
        features_file=features_file,
        split=split,
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    )

    valset.check_integrity()
    embs = valset.get_embeddings_for_class(1)
    data = valset[0]

