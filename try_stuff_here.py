from light_rl.examples.ddpg_trainer import train_ddpg
from light_rl.examples.vanilla_pg_trainer import train_vanilla_pg


# train_ddpg()
train_vanilla_pg()


'''
# summarize the sonar dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np

# load dataset
url = 'C:/Users/fadyb/Downloads/pima-indians-diabetes.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

for fn, arr in (
    ('x_train_dir.npy', X_train), ('y_train_dir.npy', y_train),
    ('x_val_dir.npy', X_val), ('y_val_dir.npy', y_val),
    ('x_test_dir.npy', X_test), ('y_test_dir.npy', y_test),
    ):
    with open(fn, 'wb') as f:
        np.save(f, arr)


import os
folder_path = ''
x_train = np.load(os.path.join(folder_path, "x_train_dir.npy"), allow_pickle=False)
y_train = np.load(os.path.join(folder_path, "y_train_dir.npy"))
x_val = np.load(os.path.join(folder_path, "x_val_dir.npy"))
y_val = np.load(os.path.join(folder_path, "y_val_dir.npy"))
x_test = np.load(os.path.join(folder_path, "x_test_dir.npy"))
y_test = np.load(os.path.join(folder_path, "y_test_dir.npy"))
print(x_train.dtype)
'''
