import bert
import tensorflow as tf
from tensorflow import keras
import config

model_dir = bert.fetch_google_albert_model(config.model_name, ".models")
model_params = bert.albert_params(config.albert_dir)


def create_model(max_seq_len, classes):
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

    input_ids = keras.layers.Input(
        shape=(max_seq_len,), dtype="int32", name="input_ids"
    )
    bert_output = l_bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    bert.load_albert_weights(
        l_bert, ".models/albert_base_v2/albert_base/model.ckpt-best"
    )

    return model
