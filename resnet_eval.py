import pandas as pd
from numpy import load
from keras.models import load_model

X_test = load("/home/mwaniki-new/Documents/deep_learning/Cats_Dogs/X_test.npy" , mmap_mode='r')

df = pd.read_csv("/home/mwaniki-new/Documents/deep_learning/Cats_Dogs/Submission/sample_submission.csv")
y_test = df['label']

model = load_model("/home/mwaniki-new/Documents/deep_learning/Cats_Dogs/src/Model/resnetmodel.h5")
resnet_model_eval = model.evaluate(X_test, y_test)

print(resnet_model_eval)