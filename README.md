A PyTorch implementation of the BI-LSTM-CRF model.

- Compared with [PyTorch BI-LSTM-CRF][1], following changes are performed:
- In the original implementation tag token indices start from '0'. '0' is also used for padding token <PAD>. This can make for erroneous training. This error is fixed by adding an appropriate unused token for padding tags sequences. 

# Installation
- dependencies
    - Python 3
    - [PyTorch][5]
- install
    ```sh
    $ pip install bi-lstm-crf
    ```

# Training
### corpus
- prepare your corpus in the specified [structure and format][2]
- there is also a sample corpus in [`bi_lstm_crf/app/sample_corpus`][3]

### training
```sh
$ python -m bi_lstm_crf corpus_dir --model_dir "model_xxx"
```
- more [options][4]
- [detail of model_dir][7]

### training curve
```python
import pandas as pd
import matplotlib.pyplot as plt

# the training losses are saved in the model_dir
df = pd.read_csv(".../model_dir/loss.csv")
df[["train_loss", "val_loss"]].ffill().plot(grid=True)
plt.show()
```


# <a id="CRF">CRF Module
The CRF module can be easily embeded into other models:
```python
from bi_lstm_crf import CRF

# a BERT-CRF model for sequence tagging
class BertCrf(nn.Module):
    def __init__(self, ...):
        ...
        self.bert = BERT(...)
        self.crf = CRF(in_features, num_tags)

    def loss(self, xs, tags):
        features, = self.bert(xs)
        masks = xs.gt(0)
        loss = self.crf.loss(features, tags, masks)
        return loss

    def forward(self, xs):
        features, = self.bert(xs)
        masks = xs.gt(0)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
```

# References
1. [Zhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirectional LSTM-CRF Models for Sequence Tagging][6]. arXiv:1508.01991.
2. PyTorch tutorial [ADVANCED: MAKING DYNAMIC DECISIONS AND THE BI-LSTM CRF][1]

[1]:https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
[2]:https://github.com/jidasheng/bi-lstm-crf/wiki/corpus-structure-and-format
[3]:https://github.com/jidasheng/bi-lstm-crf/tree/master/bi_lstm_crf/app/sample_corpus
[4]:https://github.com/jidasheng/bi-lstm-crf/wiki/training-options
[5]:https://pytorch.org/
[6]:https://arxiv.org/abs/1508.01991
[7]:https://github.com/jidasheng/bi-lstm-crf/wiki/details-of-model_dir


How to train: ICEWS
python -m bi_lstm_crf bi_lstm_crf/app/icews_corpus

Prediction: (From '/home/sudo777/EE/' directory)
python bi_lstm_crf/predict.py --sentence "to condemn the barbaric killing of farkhunda and call for justice, thousands of people from all walks of life staged a peaceful demonstration in front of the afghanistan supreme court on tuesday, demanding the apex court bring to justice all those involved in the brutal murder of the girl." --model_dir model_dir/


TWITCH 
python -m bi_lstm_crf bi_lstm_crf/app/twitch_corpus --model_dir twitch_model --num_epoch 40

python bi_lstm_crf/predict.py --model_dir twitch_model
