pip install tensorflow==2.15

import os
import shutil

def split_specific_dataset(source_dir, target_dir, splits):
    
    for class_name, counts in splits.items():
        
        class_dir = os.path.join(source_dir, class_name)
        files = sorted(os.listdir(class_dir)) 
        assert len(files) == counts['total'], f"Mismatch in total count for {class_name}"
        
        train_end = counts['train']
        val_end = train_end + counts['val']

        # Split the files into training, validation, and test sets
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
# Define the splits for each class with the number of files for training, validation, test, and total counts
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

# Load the InceptionV3 model with pretrained ImageNet weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False 
#Adding layers to modify the pre-trained model for our task
x = GlobalAveragePooling2D()(base_model.output)
x=Dense(1024,activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

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

    # Function to compute the cyclic learning rate
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
    # Set the learning rate at the beginning of training
    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.learning_rate, self.base_lr)
        else:
            K.set_value(self.model.optimizer.learning_rate, self.clr())
    # Update the learning rate at the end of each batch
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.learning_rate, self.clr())

# Data augmentation for training images
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
    batch_size=32,  
    class_mode='categorical'  
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '/kaggle/working/plantvillageapplecolor/val', 
    target_size=(299, 299),  
    batch_size=32,
    class_mode='categorical'
)

optimizer = RMSprop(0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

clr = CyclicLR(base_lr=0.00001, max_lr=0.00006, step_size=200)
early_stopping = EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)

model.fit(train_generator,
          steps_per_epoch=len(train_generator),  
          epochs=75,
          validation_data=validation_generator,
          validation_steps=len(validation_generator), 
          callbacks=[clr])
model.save('/kaggle/working/saved.h5')


# Evaluation results and Classification report

from tensorflow.keras.models import load_model

loaded_model = load_model('/kaggle/working/saved.h5')
loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
evaluation_results = loaded_model.evaluate(validation_generator)

print("Evaluation Results:")
print(f"Loss: {evaluation_results[0]}")
print(f"Accuracy: {evaluation_results[1]}")

from sklearn.metrics import classification_report, confusion_matrix

y_pred = loaded_model.predict(validation_generator)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_labels))

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

loaded_model = load_model('/kaggle/working/saved.h5')

loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize the pixel values between 0 and 1

test_generator = test_datagen.flow_from_directory(
    '/kaggle/working/plantvillageapplecolor/test',
    target_size=(299, 299),  
    batch_size=32,  
    class_mode='categorical',  
    shuffle=False  
)
print("Class indices:", test_generator.class_indices)
num_runs = 10
accuracy_list = []
sensitivity_list = []
precision_list = []
f1_score_list = []

for _ in range(num_runs):
    # Evaluate the model on the test data
    evaluation_results = loaded_model.evaluate(test_generator)
    accuracy = evaluation_results[1]
    accuracy_list.append(accuracy)

    y_pred = loaded_model.predict(test_generator)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    classification_rep = classification_report(y_true_labels, y_pred_labels, target_names=class_labels, output_dict=True)
    sensitivity = classification_rep['weighted avg']['recall']
    precision = classification_rep['weighted avg']['precision']
    f1_score = classification_rep['weighted avg']['f1-score']

    sensitivity_list.append(sensitivity)
    precision_list.append(precision)
    f1_score_list.append(f1_score)

average_accuracy = np.mean(accuracy_list)
average_sensitivity = np.mean(sensitivity_list)
average_precision = np.mean(precision_list)
average_f1_score = np.mean(f1_score_list)

print(f"Average Accuracy: {average_accuracy}")
print(f"Average Sensitivity: {average_sensitivity}")
print(f"Average Precision: {average_precision}")
print(f"Average F1 Score: {average_f1_score}")
