import tensorflow as tf
import numpy as np
from tensorflow import keras

# keras.Sequential(인자값)
# m=keras.layers.Dense(units=출력노드수, input_shape=입력형태)
# xs, ys(예제 수식은 y=2x-1)
# m.compile(손실함수, 옵티마이저함수)
# m.fit(xs, ys, epochs=반복횟수)
# m.predict(테스트값)

# 손실함수
# 1개 입력에 2개의 출력노드드를 가진 신경망에서 데이터 훈련하고 MSE 계산 연습

# 옵티마이저 알고리즘 => evaluate() 함수로 손실값을 구함
# SGD
# RSMprop
# ADAM

simple_model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

# model compile
simple_model.compile(loss='mean_squared_error', optimizer='sgd')

# fit
simple_model.fit(xs, ys, epochs=300)

# test sample
xt = [5.0]
result = simple_model.predict(xt)
print(result)