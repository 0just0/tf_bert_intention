import datetime

import bert
import sentencepiece as spm
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras

import model
import preprocess

spm_model = "../.models/albert_base_v2/albert_base/30k-clean.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model)
do_lower_case = True

classes = preprocess.CLASSES

data = preprocess.IntentDetectionData(
    preprocess.train, preprocess.test, sp, classes, max_seq_len=128
)


model = model.create_model(data.max_seq_len, classes)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

log_dir = "../log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    x=data.train_x,
    y=data.train_y,
    validation_split=0.1,
    batch_size=16,
    shuffle=True,
    epochs=5,
    callbacks=[tensorboard_callback],
)

model.save("saved_model/intent")

y_pred = model.predict(data.test_x).argmax(axis=-1)
print(classification_report(data.test_y, y_pred, target_names=classes))
