pip install tensorflow==2.15

import os
import shutil

def split_specific_dataset(source_dir, target_dir, splits):
    
    for class_name, counts in splits.items():
        class_dir = os.path.join(source_dir, class_name)
        files = sorted(os.listdir(class_dir)) 
        assert len(files) == counts['total'], f"Mismatch in total count for {class_name}"
        
        # Define split points
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

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, MaxPooling2D

# Load the base InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Freeze the layers

# Add new layers for the specific task
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x=Dense(1024,activation='relu')(x)
predictions = Dense(4, activation='softmax', kernel_regularizer=l2(0.01))(x)

model = Model(inputs=base_model.input, outputs=predictions)

import numpy as np

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CyclicLR(Callback):
    def __init__(self, base_lr=0.0001, max_lr=0.0006, step_size=200, mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.learning_rate, self.base_lr)
        else:
            K.set_value(self.model.optimizer.learning_rate, self.clr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.learning_rate, self.clr())

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


train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/plantvillageapplecolor/train',  
    target_size=(299, 299),  # Resize images to 299x299 for Inception-V3
    batch_size=64,  
    class_mode='categorical'  
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '/kaggle/working/plantvillageapplecolor/val', 
    target_size=(299, 299),  
    batch_size=16,
    class_mode='categorical'
)

optimizer = RMSprop()
def set_trainable_layers(model, unfrozen_layers):
    if unfrozen_layers == 'all':
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in model.layers[:-unfrozen_layers]:
            layer.trainable = False
        for layer in model.layers[-unfrozen_layers:]:
            layer.trainable = True
experiments = {
    'Sev_frozen': 7,
    'Ten_frozen': 10
}
all_histories = {
    'Sev_frozen': None, 
    'Ten_frozen': None,
    'none_frozen':None
}

for exp_name, unfrozen_layers in experiments.items():
    # Reset the model to be unfrozen
    set_trainable_layers(base_model, unfrozen_layers)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_generator,
          steps_per_epoch=len(train_generator),  
          epochs=50,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))
    all_histories[exp_name] = history
history_none=model.fit(train_generator,
          steps_per_epoch=len(train_generator),  
          epochs=50,
          validation_data=validation_generator,
          validation_steps=len(validation_generator),  
          callbacks=[clr, early_stopping]
                      )
all_histories['none_frozen']=history_none

print(all_histories['Sev_frozen'])

pip install matplotlib

import matplotlib.pyplot as plt

def plot_validation_accuracy(history_sev, history_ten, history_none):
    epochs = range(1, len(all_histories['Sev_frozen'].history['val_accuracy']) + 1)
    
    sev_frozen_val_accuracy = all_histories['Sev_frozen'].history['val_accuracy']
    ten_frozen_val_accuracy = all_histories['Ten_frozen'].history['val_accuracy']
    none_frozen_val_accuracy = all_histories['none_frozen'].history['val_accuracy']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sev_frozen_val_accuracy, label='Sev_frozen', marker='o')
    plt.plot(epochs, ten_frozen_val_accuracy, label='Ten_frozen', marker='o')
    plt.plot(epochs, none_frozen_val_accuracy, label='None_frozen', marker='o')
    
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_validation_accuracy(all_histories['Sev_frozen'],all_histories['Ten_frozen'],all_histories['none_frozen'])
