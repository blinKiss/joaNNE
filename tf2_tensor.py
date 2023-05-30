import tensorflow as tf
from tensorflow import keras
import numpy as np

simple_model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

# training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

# model compile
simple_model.compile(loss='mean_squared_error', optimizer='sgd')

# fit
simple_model.fit(xs, ys, epochs=500)

# test sample
xt = [5.0]
result = simple_model.predict(xt)
print(result)