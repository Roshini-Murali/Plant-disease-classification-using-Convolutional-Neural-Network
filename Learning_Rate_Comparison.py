import os
import shutil

def split_specific_dataset(source_dir, target_dir, splits):
    for class_name, counts in splits.items():
        class_dir = os.path.join(source_dir, class_name)
        files = sorted(os.listdir(class_dir))  
        assert len(files) == counts['total'], f"Mismatch in total count for {class_name}"
        
        train_end = counts['train']
        val_end = train_end + counts['val']

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        def copy_files(files, dest):
            os.makedirs(dest, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(dest, file))
        
        copy_files(train_files, os.path.join(target_dir, 'train', class_name))
        copy_files(val_files, os.path.join(target_dir, 'val', class_name))
        copy_files(test_files, os.path.join(target_dir, 'test', class_name))

source_dir = '/kaggle/input/plantvillageapplecolor'
target_dir = '/kaggle/working/plantvillageapplecolor'
splits = {
    'Apple___Black_rot': {'train': 497, 'val': 62, 'test': 62, 'total': 621},
    'Apple___Cedar_apple_rust': {'train': 221, 'val': 27, 'test': 27, 'total': 275},
    'Apple___Apple_scab': {'train': 504, 'val': 63, 'test': 63, 'total': 630},
    'Apple___healthy': {'train': 1317, 'val': 164, 'test': 164, 'total': 1645}
    }

split_specific_dataset(source_dir, target_dir, splits)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, EarlyStopping
import math
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

epochs = 50
initial_learning_rate = 1e-3

train_dir = '/kaggle/working/plantvillageapplecolor/train'
val_dir = '/kaggle/working/plantvillageapplecolor/val'

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.1,  
    zoom_range=0.1, 
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    fill_mode='nearest',  
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)

# Build InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False 

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x=Dense(1024,activation='relu')(x)
predictions = Dense(4, activation='softmax', kernel_regularizer=l2(0.01))(x)
model = Model(inputs=base_model.input, outputs=predictions)
early_stopping = EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)

def constant_lr(epoch, lr):
    return initial_learning_rate

def decay_lr(epoch, lr):
    decay_rate = 1e-1
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def clr_schedule(epoch, lr):
    base_lr = 0.0001
    max_lr = 0.0006
    step_size = 200
    cycle = math.floor(1 + epoch / (2 * step_size))
    x = abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))


constant_lr_callback = LearningRateScheduler(constant_lr)
decay_lr_callback = LearningRateScheduler(decay_lr)
clr_callback = LearningRateScheduler(clr_schedule)
optimizer_decay_lr = RMSprop(learning_rate=initial_learning_rate) 


histories = {}
# Train the model with different learning rate strategies
for lr_strategy, callback in zip(['Constant LR', 'Decay LR', 'CLR'],
                                [constant_lr_callback, decay_lr_callback, clr_callback]):
    if lr_strategy == 'Decay LR':
        model.compile(optimizer=optimizer_decay_lr,  
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='rmsprop',  
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        callbacks=[early_stopping, callback]
                       )
    histories[lr_strategy] = history

# Plotting
plt.figure(figsize=(14, 5))
strategies = ['Constant LR', 'Decay LR', 'CLR']
colors = ['blue', 'orange', 'green']

for strategy, color in zip(strategies, colors):
    plt.plot(histories[strategy].history['val_accuracy'], label=strategy, color=color)

plt.title('Learning Rate Strategy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
