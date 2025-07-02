import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

# Define Autoencoder model (must match training definition)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1 , 64  , (3,3) , stride=1 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 , 128  , (3,3) , stride=1 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128 , 256  , (3,3) , stride=1 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256 , 512  , (3,3) , stride=1 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512 , 1024  , (3,3) , stride=1 , padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2)
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(1024 , 512 , (3,3), stride=2 , padding=1 , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512 , 256 , (3,3), stride=2 , padding=1 , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256 , 128 , (3,3), stride=2 , padding=1 , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128 , 64 , (3,3), stride=2 , padding=1 , output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 3 , (3,3), stride=2 , padding=1 , output_padding=1),
            nn.Sigmoid()
        )

    def forward(self , x):
        return self.Decoder(self.Encoder(x))


# Image preprocessing
def preprocess_image(uploaded_image, img_size=128):
    image = Image.open(uploaded_image).convert("L")
    image = image.resize((img_size, img_size))
    img_np = np.array(image).astype('float32') / 255.0
    img_np = img_np.reshape(1, 1, img_size, img_size)
    return torch.tensor(img_np, dtype=torch.float32), image

# Postprocess tensor output to displayable RGB image
def tensor_to_image(tensor):
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# Load model
@st.cache_resource
def load_model():
    model = Autoencoder()
    model.load_state_dict(torch.load("colorization_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit app layout
st.title("Grayscale Image Colorization")
st.write("Upload a grayscale or color image, and the model will colorize it!")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    model = load_model()

    # Preprocess input image
    input_tensor, gray_image = preprocess_image(uploaded_file)

    with torch.no_grad():
        output = model(input_tensor)

    colorized_image = tensor_to_image(output)

    # Show input and output
    col1, col2 = st.columns(2)
    with col1:
        st.image(gray_image, caption="Grayscale Input", use_column_width=True)
    with col2:
        st.image(colorized_image, caption="Colorized Output", use_column_width=True)
