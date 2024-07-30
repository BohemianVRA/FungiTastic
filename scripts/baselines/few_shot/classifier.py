from pathlib import Path

import torch
import pandas as pd
import json
import numpy as np

from fgvc.core.metrics import classification_scores
from tqdm import tqdm
import faiss


"""
Classes only used for evaluation of the models.

"""


class Classifier(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # add option to save the results for further processing
        self.test_results = {}

    def make_prediction(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def evaluate(self, dataloader, fast_dev_run=False):
        self.eval()

        with torch.no_grad():
            im_names, predictions, gts, confidences, is_correct, probabilities = [], [], [], [], [], []
            for batch_idx, (embeddings, labels, file_paths) in tqdm(enumerate(dataloader)):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                # set ims and labels to device
                if fast_dev_run and fast_dev_run == batch_idx:
                    break
                im_names.extend([Path(fp).name for fp in file_paths])
                gts.extend(labels.cpu().numpy())
                cls, conf, probs = self.make_prediction(embeddings, ret_probs=True)
                probabilities.extend(probs.cpu().numpy() if probs is not None else [])
                predictions.extend(cls.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
                is_correct.extend((cls == labels).cpu().numpy())

        cls_scores = classification_scores(np.array(predictions) if len(probabilities) == 0 else np.array(probabilities),
                                           np.array(gts),
                                           top_k=3 if probs is not None else 1)

        print(cls_scores)

        self.test_results = {
            'im_name': im_names,
            'gt': gts,
            'pred': predictions,
            'conf': confidences,
            'is_correct': is_correct,
            'top1_acc': cls_scores['Accuracy'],
            'f1': cls_scores['F1'],
        }

        if probs is not None:
            self.test_results['top3_acc'] = cls_scores['Recall@3']
        self.test_metrics = cls_scores

    def save_results(self, out_dir, file_name=None):
        # create out dir if it doesn't exist
        out_dir.mkdir(parents=True, exist_ok=True)

        #  save test_results as dataframe
        df = pd.DataFrame(self.test_results)
        out_file = Path(out_dir) / f'{file_name}.csv'
        df.to_csv(out_file, index=False)

        # also save the test_metrics as a separate file
        out_file = Path(out_dir) / f'{file_name}_metrics.json'
        with open(out_file, 'w') as f:
            json.dump(self.test_metrics, f)


class PrototypeClassifier(Classifier):
    def __init__(self, cfg, train_embeddings, device='cuda'):
        """
        :param cfg: OmegaConf config object
        :param train_embeddings: list of C torch arrays of shape [N_C, D] where N_C is the number of training samples
        of class C and D is the dimensionality of the embeddings
        """
        super().__init__(cfg, device=device)

        # C x D array of class prototypes, make them a parameter so that they are moved to the device
        self.class_prototypes = self.get_prototypes(train_embeddings, mode=cfg.classifier)
        self.class_prototypes = torch.nn.Parameter(self.class_prototypes, requires_grad=False)

    def get_prototypes(self, embeddings, mode='centroid'):
        if mode == 'centroid':
            class_prototypes = torch.stack([class_embs.mean(dim=0) for class_embs in embeddings])
        elif mode == 'one_shot':
            # take the first occurence of each class as the prototype, and classify based on the closest prototype
            class_prototypes = embeddings[:, 0]
        return class_prototypes

    def make_prediction_dist(self, embeddings):
        """

        :param embeddings: torch.Tensor of shape (batch_size, n_channels, height, width)
        :return: probabilities of shape (batch_size, n_classes) computed based on
        the similarity of the embeddings to the class prototypes
        """

        # compute the similarity of each embedding to each prototype
        # embeddings - [N, D], class_prototypes - [C, D]
        dists = torch.cdist(embeddings, self.class_prototypes)
        # get the class with the smallest distance
        cls = torch.argmin(dists, dim=1)
        # get the confidence of the prediction
        conf = 1 - dists[torch.arange(embeddings.shape[0]), cls]
        return cls, conf

    def make_prediction(self, embeddings, plot_sim_hist=False, ret_probs=False):
        # compute the cosine similarity of each embedding to each prototype
        similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), self.class_prototypes.unsqueeze(0), dim=-1)
        # get the class with the highest similarity
        cls = torch.argmax(similarities, dim=1)
        probs = torch.nn.functional.softmax(similarities, dim=1)
        # get the confidence of the prediction from softmax
        conf = probs.max(dim=1).values
        if plot_sim_hist:
            import matplotlib.pyplot as plt
            plt.hist(similarities[1].cpu().numpy(), bins=100)
            plt.show()
        if ret_probs:
            return cls, conf, probs
        else:
            return cls, conf


class NNClassifier(Classifier):
    def __init__(self, cfg, train_embeddings, device='cuda'):
        """
        :param cfg: config object, namespace
        :param train_embeddings: list of C torch arrays of shape [N_C, D] where N_C is the number of training samples
        of class C and D is the dimensionality of the embeddings
        """
        super().__init__(cfg, device=device)

        self.index, self.idx2cls = self.build_index(train_embeddings)

    def make_prediction(self, embeddings, plot_sim_hist=False, ret_probs=False):
        """
        :param embeddings: torch.Tensor of shape (batch_size, n_channels, height, width)
        :return: probabilities of shape (batch_size, n_classes) computed based on
        the similarity of the embeddings to the class prototypes
        """
        # compute the similarity of each embedding to each prototype
        # embeddings - [N, D], class_prototypes - [C, D]
        similarities, indices = self.index.search(embeddings.cpu().numpy(),1)
        # get the classes for the indices
        cls = self.idx2cls[indices.squeeze()]
        # get the confidence of the prediction
        conf = similarities
        probs = None
        return cls, conf, probs

    def build_index(self, train_embeddings):
        idx2cls = np.hstack([np.ones(len(embs)) * i for i, embs in enumerate(train_embeddings)])
        # concatenate the embeddings
        embs = torch.cat(train_embeddings)
        # build the index for cosine similarity search
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.cpu().numpy())
        return index, idx2cls

    def evaluate(self, dataloader, fast_dev_run=False):
        self.eval()

        with torch.no_grad():
            im_names, predictions, gts, confidences, is_correct, probabilities = [], [], [], [], [], []
            for batch_idx, (embeddings, labels, file_paths) in tqdm(enumerate(dataloader)):
                # set ims and labels to device
                if fast_dev_run and fast_dev_run == batch_idx:
                    break
                im_names.extend([Path(fp).name for fp in file_paths])
                gts.extend(labels)
                cls, conf, probs = self.make_prediction(embeddings, ret_probs=True)
                probabilities.extend(probs if probs is not None else [])
                predictions.extend(cls)
                confidences.extend(conf)
                is_correct.extend((cls == labels.numpy()))

        cls_scores = classification_scores(np.array(predictions) if len(probabilities) == 0 else np.array(probabilities),
                                           np.array(gts),
                                           top_k=3 if probs is not None else 1)

        print(cls_scores)

        self.test_results = {
            'im_name': im_names,
            'gt': gts,
            'pred': predictions,
            'conf': confidences,
            'is_correct': is_correct,
            'top1_acc': cls_scores['Accuracy'],
            'f1': cls_scores['F1'],
        }

        if probs is not None:
            self.test_results['top3_acc'] = cls_scores['Recall@3']
        self.test_metrics = cls_scores