import os
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')
data_dir = os.path.join(path, 'plantvillage dataset', 'color')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

valid_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)

data_augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])
normalize = layers.Rescaling(1./255)

def prepare_train(x, y):
    x = data_augment(x, training=True)
    x = normalize(x)
    return x, y

def prepare_valid(x, y):
    x = normalize(x)
    return x, y

train_ds = (
    train_ds_raw
    .map(prepare_train, num_parallel_calls=AUTOTUNE)
    .shuffle(200)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    valid_ds_raw
    .map(prepare_valid, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'plant_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=valid_ds,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

model.save('plant_model.keras')

base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    epochs=20,
    initial_epoch=10,
    validation_data=valid_ds,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

model.save('plant_model_finetuned.keras')

y_true = []
y_pred = []
for images, labels in valid_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Plant Disease Classification')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
