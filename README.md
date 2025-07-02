# 🎨 Image Colorization with Convolutional Autoencoder

Colorizing grayscale images using a deep convolutional autoencoder built with PyTorch. This project was trained on the CelebA dataset and achieves over **75% pixel-wise accuracy** in reconstructing color from grayscale input.

---

## 🚀 Live Demo

👉 [Open Streamlit App](https://colorized-autoencoder-wpathkvbyapehdff6kpz2k.streamlit.app/)

Upload a grayscale image and watch it come to life with colors!

---

## 🧠 Model Architecture

- **Encoder**: 5 convolutional blocks with ReLU + BatchNorm + MaxPooling
- **Decoder**: 5 transposed convolutional layers for upsampling
- **Input**: 1-channel grayscale image (128x128)
- **Output**: 3-channel RGB image (128x128)

Trained using MSE Loss and Adam Optimizer.

---

## 🧪 Results

| Grayscale Input  Colorized Output |
|-----------------------------------|
|  ![Screenshot 2025-07-02 191417](https://github.com/user-attachments/assets/297c3c0d-0e91-4a53-93be-c2f3aa757bdc)
 |


📈 **Accuracy**: `~75% pixel-level accuracy` (tolerance = 0.05)

---

## 🛠️ How to Run Locally

### 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

Upload any grayscale image (128x128 recommended) to see the colorization results.

---

## 🧾 Project Structure

```
├── app.py                     # Streamlit app
├── model.py                   # Autoencoder model definition
├── convert_to_grayscale.py    # Script to convert RGB to grayscale
├── colorization_model.pth     # Trained model weights
├── requirements.txt
├── README.md
└── examples/
    ├── input.jpg
    └── output.jpg
```

---

## 📁 Dataset

- Used the **CelebA dataset** from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- Trained on 60,000 facial images
- Preprocessed to 128x128 grayscale and RGB

---

## 🧠 Training Details

- **Platform**: Google Colab
- **Optimizer**: Adam
- **Loss**: MSELoss
- **Epochs**: 25
- **Batch Size**: 64

Training visualization included below:

---

## ✨ Highlights

- ✅ 75%+ accuracy
- ✅ Lightweight autoencoder architecture
- ✅ Live Streamlit web app
- ✅ Easy to train and extend

---

## 📬 Contact

If you have questions or want to collaborate:

- GitHub: [@Junaid0-glitch](https://github.com/Junaid0-glitch)
- Email: `junaid76170@gmail.com`

---

## ⭐ Give it a Star!

If you liked the project, please consider ⭐ starring the repo — it helps others discover it too!
