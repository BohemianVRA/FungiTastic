# 🍄 FungiCLEF 2025 Benchmark

**FungiCLEF 2025** is the fourth edition of the international challenge for *few-shot fungi species recognition* and is part of the [LifeCLEF Lab](https://www.imageclef.org/LifeCLEF2025) and [FGVC Workshop at CVPR](https://sites.google.com/view/fgvc12/).
In 2025, the benchmark advances **few-shot learning** for rare wild fungi, where all classes have only 1–4 labeled examples.

**See the official [Kaggle competition](https://www.kaggle.com/competitions/fungi-clef-2025/overview) for full details, data, and results.**

---

## 🏆 Leaderboard

| Place      | Top-5 Accuracy (%) | Team Name            | Paper / Link                                                                                                                    |
|------------| ------------------ | -------------------- |---------------------------------------------------------------------------------------------------------------------------------|
| **1st**    | 78.9               | Jack Etheredge       | *Few-Shot Fungi Classification with Prototypical Networks Using Multiple Pretrained Embedding Models* <br>*(To be linked)*      |
| **2nd**    | 78.1               | hard\_work           | *Few-Shot Fine-Grained Classification of Fungi Species Using Contrastive Representation Learning* <br>*(To be linked)*          |
| **17th**   | 60.3               | Embia                | *Improving Fungi Prototype Representations for Few-Shot Classification* <br>*(To be linked)*                                    |
| **22nd**   | 57.4               | I2C-UHU-Pegasus      | *Multi-Modal Pipeline for Rare Fungal Species Classification using Fine-tuned VLMs and Ecological Context* <br>*(To be linked)* |
| **26th**   | 55.5               | Yang Tuấn Anh        | *Mushroom for Improvement: Prototypical Few-Shot Learning with Multimodal Fungal Features* <br>*(To be linked)*                 |
| **35th**   | 46.4               | DS\@GT LifeCLEF      | *Transfer Learning and Mixup for Fine-Grained Few-Shot Fungi Classification* <br>*(To be linked)*                               |
| _Baseline_ | 26.7               | Prototype Classifier | [FungiTastic: A Multi-Modal Dataset and Benchmark for Image Categorization](https://arxiv.org/pdf/2408.13632)                                                                           |
| _Baseline_ | 24.7               | Nearest Neighbor     | [FungiTastic: A Multi-Modal Dataset and Benchmark for Image Categorization](https://arxiv.org/pdf/2408.13632)                     |

*The top submission outperformed all baselines by a wide margin! See the [Kaggle leaderboard](https://www.kaggle.com/competitions/fungi-clef-2025/overview) for full results.*


## 📄 References

```
@inproceedings{fungiclef2025,
    author={Janouskova, Klara and Matas, Jiri and Picek, Lukas},
    title = {Overview of {FungiCLEF} 2025: Few-Shot Classification With Rare Fungi Species},
    booktitle={Working Notes of CLEF 2025 - Conference and Labs of the Evaluation Forum},
    year={2025}
}

@inproceedings{lifeclef2025,
  title={Overview of LifeCLEF 2025: Challenges on Species Presence Prediction and Identification, and Individual Animal Identification},
  author={Picek, Lukas and Kahl, Stefan and Go{\"e}au, Herv{\'e} and Adam, Luk{\'a}{\v{s}} and Larcher, Th{\'e}o and Leblanc, Cesar and Servajean, Maximilien and Janou{\v{s}}kov{\'a}, Kl{\'a}ra and Matas, Ji{\v{r}}{\'\i} and {\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Papafitsoros, Kostas and Planqu{\'e}, Robert and Vellinga, Willem-Pier and Klinck, Holger and Denton, Tom and Ca{\~n}as, Juan Sebasti{\'a}n and Martellucci, Giulio and Vinatier, Fabrice and Bonnet, Pierre and Joly, Alexis},
  booktitle={International Conference of the Cross-Language Evaluation Forum for European Languages (CLEF)},
  year={2025},
  organization={Springer}
}
```
