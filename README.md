# NSE2Vec 📈

Please leave a ⭐ if you like the repository.

This repository contains code for reproducing the results of [Stock Embeddings: Learning Distributed Representations for Financial Assets](https://arxiv.org/pdf/2202.08968.pdf) by [Rian Dolphin](https://github.com/rian-dolphin) et al on Indian Stock Market Data (NSE). The code is largely borrowed from [rian-dolphin/stock-embeddings](https://github.com/rian-dolphin/stock-embeddings).

Please follow [main.ipynb](main.ipynb) for the code to train the embedding model. Supporting classes are in the [utils/](utils/) folder.

### Directory Structure
```shell
.
├── README.md
├── data
│   ├── data-utilities.ipynb
│   ├── historical-stocks.csv
│   ├── nse-data.csv
│   ├── nsenbse-data.csv
│   └── scrape-industry-data.ipynb
├── main.ipynb
├── nse2vec.pt
└── utils
    ├── __init__.py
    ├── evaluate.py
    ├── format_data.py
    ├── model.py
    ├── stock_data.py
    ├── train.py
    └── visualize.py
```

### Final Results:
| Data | Train Epochs | Accuracy | Accuracy (top 3) |
| --- | --- | --- | --- |
| NSE | 5 | 0.75 | 0.9 |
| NSE+BSE | 5 | 0.76 | 0.9 |

### Original Paper:
```
@article{dolphin2022stock,
  title={Stock embeddings: Learning distributed representations for financial assets},
  author={Dolphin, Rian and Smyth, Barry and Dong, Ruihai},
  journal={arXiv preprint arXiv:2202.08968},
  year={2022}
}
```