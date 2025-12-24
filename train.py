import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers


DATA_DIR = 'dataset_baru/train' 
IMG_SIZE = (160, 160)
BATCH_SIZE = 32


train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"\nKelas yang ditemukan: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.1)(x)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) 
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class_weights = {}
total_files = 0
class_counts = []
for i, name in enumerate(class_names):
    folder_path = os.path.join(DATA_DIR, name)
    count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    class_counts.append(count)
    total_files += count

for i, count in enumerate(class_counts):
    weight = total_files / (len(class_names) * count)
    class_weights[i] = weight

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'skin_model.keras', 
    save_best_only=True,  
    monitor='val_accuracy', 
    mode='max'
)
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=validation_dataset,class_weight = class_weights,callbacks=[checkpoint])

model.save('skin_model.keras')

with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("Selesai!")