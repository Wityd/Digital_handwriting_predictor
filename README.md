# ✏️ Digit Classifier

A handwritten digit recognition web app built with **scikit-learn** and **Streamlit**. Upload an image of a handwritten digit (0–9) and the model predicts what it is, along with a confidence score for every class.

🔗 **Live app:** [Streamlit Cloud link here]

---

## 📊 Overview

| | |
|---|---|
| **Model** | Random Forest Classifier (`n_estimators=20`) |
| **Dataset** | [MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 training images, 10,000 test images |
| **Input** | 28×28 grayscale image, flattened to a 784-feature vector |
| **Test accuracy** | ~95.8% |

---

## 🧠 How it works

1. **Training** (`train_and_save.py`) — loads the MNIST `.gz` files, trains a `RandomForestClassifier`, evaluates it on the test set, and saves the trained model to `model.pkl` using `joblib`.
2. **Inference** (`app.py`) — a Streamlit interface where a user uploads an image. The image is preprocessed (grayscaled, auto-inverted if needed, cropped to the digit, centered, resized to 28×28) and passed to the model for prediction.

---

## 🖼️ Image preprocessing pipeline

Real-world images don't look like raw MNIST data, so `app.py` applies a few steps to bridge the gap:

1. Convert to grayscale
2. Auto-detect background brightness and invert colors if needed, so the digit is always light-on-dark (matching MNIST format)
3. Detect the digit using the largest connected region of bright pixels (ignoring small noise specks/artifacts)
4. Crop tightly to the digit's bounding box
5. Pad to a square with a small margin and center the digit (matching MNIST's centered layout)
6. Resize to 28×28 pixels

---

## ⚠️ Known limitation

MNIST digits are **thick, blocky, and drawn by mouse/stylus** on a small canvas. Digits from phone screenshots, fonts, or smooth digital drawings tend to be **thinner, anti-aliased, and stylistically different** from MNIST's training distribution. Because the model has only ever seen MNIST-style strokes, accuracy can drop noticeably on these inputs even after preprocessing.

**For best results:**
- Use a thick, bold, hand-drawn digit
- Make sure the digit is on a plain background (light or dark — the app auto-detects)
- Avoid very thin or stylized fonts

This is a dataset/domain limitation, not a bug — a model is only as good as the data it's trained on.

---

## 🚀 Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needs to be run once)
python train_and_save.py

# 3. Launch the app
streamlit run app.py
```

Make sure your MNIST `.gz` files live in a folder called `datasets/` in the project root before running step 2.

---

## 📁 Project structure

```
.
├── app.py                # Streamlit web app
├── train_and_save.py     # Trains the model and saves model.pkl
├── model.pkl             # Saved trained model (generated)
├── requirements.txt      # Python dependencies
├── datasets/             # MNIST .gz files
└── Image_classification.ipynb   # Original notebook (exploration/training)
```

---

## 🛠️ Tech stack

- **Python**
- **scikit-learn** — Random Forest model
- **Streamlit** — web app framework
- **Pillow / NumPy / SciPy** — image preprocessing

---

## 📌 Possible future improvements

- Train on a more diverse dataset (e.g. augmented MNIST, or real-world digit photos) to close the domain gap
- Try a CNN instead of Random Forest for better accuracy on varied input styles
- Add a drawing canvas directly in the app instead of requiring file upload
