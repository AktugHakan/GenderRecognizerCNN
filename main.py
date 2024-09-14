import tensorflow as tf
import cv2
import random
from image_importer import load_dataset, shuffle_dataset
keras = tf.keras
layers = keras.layers
from pathlib import Path
import numpy as np

def show_image(x,y,idx = None):
    if (idx is None):
        idx = random.randint(0, x.shape[0] - 1)

    cv2.imshow("MALE" if y[idx] == 1.0 else "FEMALE", x[idx])
    cv2.waitKey(0)

def predict_image(x,y,model,idx=None):
    if (idx is None):
        idx = random.randint(0, x.shape[0] - 1)
    prediction = model.predict(np.array([x[idx]]))
    predict_male = prediction > 0.5
    predict_correct = predict_male == (y[idx] == 1.0)
    cv2.imshow(("MALE" if predict_male else "FEMALE") + (" - T" if predict_correct else " - F" ) + f"M[%{prediction.item()*100:.2f}]", x[idx])
    cv2.waitKey(0)
    

def generate_model():
    model_cache_path = Path(".cache/model.keras")
    pretrained_model_path = Path("pretrained_model/model.keras")
    if model_cache_path.exists():
        model = keras.models.load_model(model_cache_path)
        model.summary()
    elif pretrained_model_path.exists():
        model = keras.models.load_model(pretrained_model_path)
        model.summary()
        model.save(model_cache_path)
    else:
        model = keras.models.Sequential()
        model.add(layers.Input((80,100,3)))

        model.add(layers.Conv2D(32, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        training_dir = Path("archive/Training/")
        train_X, train_Y = load_dataset(training_dir / "male", training_dir / "female")
        train_X, train_Y = shuffle_dataset(train_X, train_Y)

        train_X = train_X.astype("float32") / 255.0
        train_Y = train_Y.astype("float32")

        model.fit(train_X, train_Y, epochs=30)
        model.summary()
        model.save(model_cache_path)
        
    return model

model = generate_model()
test_dir = Path("archive/Validation/")
test_X, test_Y = load_dataset(test_dir / "male", test_dir / "female")

for i in range(50):
    predict_image(test_X, test_Y, model)