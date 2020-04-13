import bert
import tensorflow as tf
import numpy as np

import config
import preprocess

import sentencepiece as spm

pred_sentences = [
    "add sabrina salerno to the grime instrumentals playlist",
    "i want to bring four people to a place that s close to downtown that serves churrascaria cuisine",
    "put lindsey cardinale into my hillary clinton s women s history month playlist",
    "will it snow in mt on june 13  2038",
    "play signe anderson chant music that is newest",
    "can you let me know what animated movies are playing close by",
]

spm_model = "../.models/albert_base_v2/albert_base/30k-clean.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model)
do_lower_case = True

classes = preprocess.CLASSES


def _pad(ids, max_seq_len):
    x = []
    for input_ids in ids:
        input_ids = input_ids[: min(len(input_ids), max_seq_len - 2)]
        input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
        x.append(np.array(input_ids))
    return np.array(x)


processed_text = [
    bert.albert_tokenization.preprocess_text(p, lower=True) for p in pred_sentences
]
pred_token_ids = [bert.albert_tokenization.encode_ids(sp, p) for p in processed_text]

tokens_len = max([len(tokens_l) for tokens_l in pred_token_ids])
max_seq_len = max(config.MAX_SEQ_LEN, tokens_len)

pred_token_ids = _pad(pred_token_ids, max_seq_len)
print(pred_token_ids.shape)

model = tf.keras.models.load_model("../saved_model/intent")

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, intent in zip(pred_sentences, res):
    print(" text:", text)
    print("  res:", classes[intent])
