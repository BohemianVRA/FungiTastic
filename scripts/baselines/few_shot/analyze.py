"""
the dependency 'tabulate' is required to run this script with markdown output
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from dataset.fungi import FungiTastic

CLASSIFIERS = ['centroid', 'nn']
SPLITS = ['val', 'test', 'valtest']
FEATURE_MODELS = ['bioclip', 'clip', 'dinov2']


def get_results_path(base_res_dir, split, classifier, feature_model, metrics_only=True):
    if metrics_only:
        return base_res_dir / split / f"eval_fungi_{feature_model}_{split}_{classifier}_metrics.json"
    else:
        return base_res_dir / split / f"eval_fungi_{feature_model}_{split}_{classifier}.csv"


def load_result(path):
    ext = path.suffix
    if ext == '.csv':
        df = pd.read_csv(path)
        # map gt ('tensor(1010)') and pred ('444.0') to integers - TODO this should be fixed in the evaluation script
        # if gt and pred type is not already int
        if not df['gt'].dtype == 'int':
            df['gt'] = df['gt'].map(lambda x: int(x[7:-1]))
        if not df['pred'].dtype == 'int':
            df['pred'] = df['pred'].map(lambda x: int(x))
        return df
    elif ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f'Unknown file extension: {path}')


def get_split_results(results_dir, split, show=False):
    results = []
    for feature_model in FEATURE_MODELS:
       for classifier in CLASSIFIERS:
           res_path = get_results_path(results_dir, split, classifier, feature_model)
           res = load_result(res_path)
           top3_acc = res['Recall@3'] if classifier == 'centroid' else None
           results.append([feature_model, classifier, res['Accuracy'], res['F1'], top3_acc])

    # add CE baselines
    res_path = get_results_path(results_dir, split, 'ce', 'resnet')
    res = load_result(res_path)
    top3_acc = res['Recall@3'] if classifier == 'centroid' else None
    results.append([feature_model, classifier, res['Accuracy'], res['F1'],
                    top3_acc])

    cols = ['feature_model',  'classifier', 'accuracy', 'f1', 'recall@3']
    results_df = pd.DataFrame(results, columns=cols)

    if show:
        #     group by feature model and print the grouped results
        print_df = results_df.groupby('feature_model').apply(lambda a: a[:])
        # print * 100, rounded up to two decimal places
        print_df[['accuracy', 'f1', 'recall@3']] *= 100
        print(print_df.round(2))
    return results_df


def load_all_results(results_dir):
    results = []
    for split in SPLITS:
        for feature_model in FEATURE_MODELS:
            for classifier in CLASSIFIERS:
                res_path = get_results_path(results_dir, split, classifier, feature_model)
                res = load_result(res_path)
                top3_acc = res['Recall@3'] if classifier == 'centroid' else None
                results.append([split, feature_model, classifier, res['Accuracy'], res['F1'],
                                top3_acc])

    cols = ['split', 'feature_model', 'classifier', 'accuracy', 'f1', 'recall@3']
    results_df = pd.DataFrame(results, columns=cols)
    return results_df


def overall_results_latex_table(results_dir, split):
    results_df = get_split_results(results_dir, split)
    # multiply metrics by 100
    results_df[['accuracy', 'f1', 'recall@3']] *= 100
    results_df = results_df.groupby('feature_model').apply(lambda a: a[:])
    # index needs to be used for multirow, so remove it afterwards
    print(results_df.to_latex(index=True, float_format="{:.1f}".format))


def get_accs_per_shot(train_df, results_dir, split, feature_model, classifier):
    res = load_result(get_results_path(
        base_res_dir=results_dir,
        split=split,
        classifier=classifier,
        feature_model=feature_model,
        metrics_only=False,
    ))

    # get number of observations (images with the same 'observationID') for each 'category_id' in the training set
    n_obs_per_class = train_df.groupby('category_id')['observationID'].nunique()

    max_shot_n = n_obs_per_class.max()

    # compute accuracy for each shot
    acc_per_shot = {}
    for shot_n in range(1, max_shot_n + 1):
        shot_clses = n_obs_per_class[n_obs_per_class == shot_n].index
        # filter res df by classes with shot_n observations, keep only gt and predictions
        shot_res = res[res['gt'].isin(shot_clses)][['gt', 'pred']]
        acc = (shot_res['gt'] == shot_res['pred']).mean()
        acc_per_shot[shot_n] = acc

    return acc_per_shot


def per_shots_results_plot(results_dir, data_dir, split, feature_model, classifier, ax=None, show=True, save_dir=None):
    train_df = FungiTastic(
        root=data_dir,
        split='train',
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    ).df

    acc_per_shot = get_accs_per_shot(train_df, results_dir, split, feature_model, classifier)
    acc_per_shot = pd.Series(acc_per_shot)
    acc_per_shot.plot(kind='bar', title=f'Accuracy per shot - {split}, {feature_model}-{classifier}', xlabel='Number of shots', ax=ax)

    if save_dir:
        plt.savefig(save_dir / f'{split}_{feature_model}_{classifier}_per_shot.png')
    if show:
        plt.show()


def overall_per_shot_results_table(results_dir, data_dir, split, out='md', show=True):
    train_df = FungiTastic(
        root=data_dir,
        split='train',
        size='300',
        task='closed',
        data_subset='FewShot',
        transform=None,
    ).df

    results = []
    for feature_model in FEATURE_MODELS:
        for classifier in CLASSIFIERS:
            acc_per_shot = get_accs_per_shot(train_df, results_dir, split, feature_model, classifier)
            for shot_n, acc in acc_per_shot.items():
                results.append([feature_model, classifier, shot_n, acc])

    cols = ['model', 'classifier', 'k', 'accuracy']
    results_df = pd.DataFrame(results, columns=cols)
    # multiply accuracy by 100
    results_df['accuracy'] *= 100

    if show:
        if out == 'md':
            print(results_df
                    .groupby('k')
                    .apply(lambda a: a[:])
                    .T
                    .to_markdown(index=False, floatfmt=".2f")
                    )

            # print the tabel for each shot separately, without the shots columns

            print(3 * '\n')


            for k in results_df['k'].unique():
                print(f'**{k} shot:**')
                print(results_df[results_df['k'] == k].drop(columns='k').to_markdown(index=False, floatfmt=".2f"))
                print(3 * '\n')
        elif out == 'latex':
            print(results_df.groupby('k')
                  .apply(lambda a: a[:])
                  .to_latex(index=False, float_format="{:.2f}".format)
                  )
        else:
            raise ValueError(f'Unknown output format: {out}')

    return results_df



if __name__ == '__main__':
    config_path = '../../../config/FungiTastic_FS.yaml'
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    results_dir = Path(cfg.path_out) / 'results' / 'fs'
    # res_path = get_results_path(results_dir, 'val', 'centroid', 'bioclip')
    # res = load_result(res_path)
    # print(res)
    # get_split_results(results_dir, 'valtest')
    # overall_results_latex_table(results_dir, 'valtest')

    # per_shots_results_plot(
    #     results_dir=results_dir,
    #     data_dir=cfg.data_path,
    #     split='test',
    #     feature_model='bioclip',
    #     classifier='nn',
    #     # save_dir=Path(cfg.path_out) / 'vis' / 'fs'
    #     save_dir=None
    # )

    overall_per_shot_results_table(results_dir, cfg.data_path, 'test', show=True, out='md')
