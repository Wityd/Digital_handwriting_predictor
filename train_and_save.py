"""
Run this script ONCE to train and save your model.
It reads the same MNIST .gz files your notebook uses.
Usage: python train_and_save.py
"""

import numpy as np
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

IMAGE_WIDTH = 28


def extract_data(filename, num_images, IMAGE_WIDTH):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def get_data(num_train_images, num_test_images):
    DATA_DIR = "datasets"  # same path as your notebook
    X_train = extract_data(f"{DATA_DIR}/train-images-idx3-ubyte.gz", num_train_images, IMAGE_WIDTH)
    y_train = extract_labels(f"{DATA_DIR}/train-labels-idx1-ubyte.gz", num_train_images)
    X_train /= 255.0

    X_test = extract_data(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz", num_test_images, IMAGE_WIDTH)
    y_test = extract_labels(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz", num_test_images)
    X_test /= 255.0

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = get_data(60000, 10000)

    print("Training RandomForestClassifier (n_estimators=20)...")
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to model.pkl ✅")
