import cv2 as cv
import numpy as np
import os
from pathlib import Path
from hashlib import md5

def load_dataset(male_directory, female_directory, invalidate_cache = False):
    x_final = None
    y_final = None

    cache_name_prefix = md5(str(male_directory).encode() + str(female_directory).encode()).hexdigest()
    X_cache_path = Path(f".cache/{cache_name_prefix}_X.npy")
    Y_cache_path = Path(f".cache/{cache_name_prefix}_Y.npy")
    if invalidate_cache or not X_cache_path.exists() or not Y_cache_path.exists():
        x = [] # raw image
        y = [] # is male

        for img_path in os.listdir(male_directory):
            img_raw = cv.imread(os.path.join(male_directory, img_path))
            img_raw = cv.resize(img_raw, (80,100))
            x.append(img_raw)
            y.append(True)

        for img_path in os.listdir(female_directory):
            img_raw = cv.imread(os.path.join(female_directory, img_path))
            img_raw = cv.resize(img_raw, (80,100))
            x.append(img_raw)
            y.append(False)
        
        x_final, y_final = np.array(x), np.array(y)
    else:
        x,y = (np.load(X_cache_path), np.load(Y_cache_path))
        if len(x) == len(y):
            x_final,y_final = x,y 
        else:
            print("Cache is corrupted! Trying to import from files.")
            x_final,y_final = load_dataset(male_directory, female_directory, True)

    Path(".cache").mkdir(exist_ok=True)
    np.save(X_cache_path, x_final)
    np.save(Y_cache_path, y_final)
    return x_final, y_final

def shuffle_dataset(x,y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x_new = x[indices]
    y_new = y[indices]

    return x_new, y_new