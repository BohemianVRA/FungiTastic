# Open-set classification experiments over FungiTastic

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the training scripts and data tools.
- Checkpoints are available at [Hugging Face Hub Repository](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b).


## Installation
Python 3.10+ is required.
### Local instalation
Uses the same requirements as the closed-set experiments. Just use the following commands in your terminal.
```
pip install -r requirements.txt
```


## Open-set inferences

To reproduce our results, run:
1. Extract features and logits using `extract_features.ipynb` notebook.
2. Train linear DINOv2 classifier using `dino_linear.ipynb` notebook.
3. Evaluate the scores and calculate metrics in `ood.ipynb` notebook.

