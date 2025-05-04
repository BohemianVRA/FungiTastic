# Close-set classification experiments over FungiTastic and FungiTastic-Mini 

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the training scripts and data tools.
- Checkpoints are available at [Hugging Face Hub Repository](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b).
- Train and Validation logs are available at [Weights & Biases Workspace](https://wandb.ai/zcu_cv/FungiTastic).


## Installation
Python 3.10+ is required.
### Local instalation
1. Install dependencies
You can use any virtual or local environment. Just use the following commands in your terminal.
```
pip install -r requirements.txt
```
2. Login to [Weights & Biases](https://wandb.ai/site) to log results [*optional].
```
wandb login
```
3. Login to [Hugging Face Hub](https://huggingface.co/) to save and download model checkpoints [*optional].
```
huggingface-cli login
```

## Training
To run the training you can use the provided `train.ipynb` notebook or `train.py` CLI.
In both you have to:
* Specify valid paths, wandb settings, etc. in **train.ipynb** or local environment and run. In the notebook
all variables that must be "set" have `"changethis"` as value.

```
python train.py \
    --train-path $TRAIN_METADATA_PATH \
    --test-path $TEST_METADATA_PATH \
    --config-path ../configs/DF24M_224_config.yaml \
    --cuda-devices $CUDA_DEVICES \
    --wandb-entity $WANDB_ENTITY [**optional**] \
    --wandb-project $WANDB_PROJECT [**optional**] \
    --hfhub-owner $HFHUB_OWNER [**optional**]
```


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICES"]
