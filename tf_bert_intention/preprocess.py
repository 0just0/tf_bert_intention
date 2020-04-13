import bert
import pandas as pd
from tqdm import tqdm

import numpy as np

train = pd.read_csv("../data/train.csv")
valid = pd.read_csv("../data/valid.csv")
test = pd.read_csv("../data/test.csv")

train = train.append(valid).reset_index(drop=True)

CLASSES = train.intent.unique().tolist()
MAX_SEQ_LEN = 128

print(CLASSES)


class IntentDetectionData:
    DATA_COLUMN = "text"
    LABEL_COLUMN = "intent"

    def __init__(self, train, test, tokenizer, classes, max_seq_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        self.classes = classes

        train, test = map(
            lambda df: df.reindex(
                df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index
            ),
            [train, test],
        )

        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(
            self._prepare, [train, test]
        )

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        for _, row in tqdm(df.iterrows()):
            text, label = (
                row[IntentDetectionData.DATA_COLUMN],
                row[IntentDetectionData.LABEL_COLUMN],
            )
            processed_text = bert.albert_tokenization.preprocess_text(text, lower=True)
            token_ids = bert.albert_tokenization.encode_ids(
                self.tokenizer, processed_text
            )
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[: min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)
