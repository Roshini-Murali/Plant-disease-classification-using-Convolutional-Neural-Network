from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pretrained InceptionV3 model with weights trained on ImageNet
model = InceptionV3(weights='imagenet')

# Specify the layer names and their corresponding output sizes that we want to visualize
layer_names = [
    'input_layer',
    'conv2d_1',       
    'conv2d_2',     
    'conv2d_3',     
    'conv2d_4',
    'conv2d_5',     
    'mixed0',       
    'mixed5',       
    'mixed10'       
]

sizes = [
    (299, 299),   
    (127, 127),   
    (125, 125),   
    (125, 125),   
    (62, 62),     
    (60, 60),     
    (29, 29),     
    (14, 14),     
    (6, 6)        
]

layers_output = [model.get_layer(name).output for name in layer_names]
model = Model(inputs=model.input, outputs=layers_output)

img_path = '/kaggle/input/plantvillageapplecolor/Apple___Black_rot/01e94c43-0879-4e8c-9b61-c48cfed88dab___JR_FrgE.S 3024.JPG'  
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

activations = model.predict(x)

for layer_activation, layer_name, size in zip(activations, layer_names, sizes):
    n_features = layer_activation.shape[-1]  
    size_x, size_y = size 

    n_cols = n_features // 16  
    display_grid = np.zeros((size_x * n_cols, size_y * 16))

    for col in range(n_cols):
        for row in range(16):
            channel_image = layer_activation[0, :, :, col * 16 + row]
            channel_image -= channel_image.mean() 
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            
            channel_image = np.array(Image.fromarray(channel_image).resize((size_y, size_x)))
            
            display_grid[col * size_x : (col + 1) * size_x, row * size_y : (row + 1) * size_y] = channel_image

    scale = 1. / size_x
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(f'{layer_name} - {size_x} x {size_y}')
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
