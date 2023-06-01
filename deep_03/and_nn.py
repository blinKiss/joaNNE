import tensorflow as tf
from tensorflow import keras
import numpy as np

model_and = keras.Sequential([
    keras.layers.Dense(units=3, input_shape=[2], activation='relu'),
    keras.layers.Dense(units=1)
    ])

tf.random.set_seed(0)

# 학습 데이터
x_train = [ [0,0], [0,1], [1,0], [1,1] ]
y_train = [ [0], [0], [0], [1] ]

# 컴파일
model_and.compile(loss='mse', optimizer='adam')

result_before = model_and.predict(x_train)
print('훈련 전\n', result_before)

# epochs 횟수 훈련
loss_history = model_and.fit(x_train, y_train, epochs=1000, verbose=0)

result_after = model_and.predict(x_train)
print('훈련 후\n', result_after)

# 손실값의 변화를 그래프로
import matplotlib.pyplot as plt

loss = loss_history.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()




