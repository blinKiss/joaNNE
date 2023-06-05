import tensorflow as tf
import numpy as np
from tensorflow import keras

# 랜덤하게 가중치 설정
tf.random.set_seed(0)

# 모델을 만들기 전에 뉴런층을 정의 
input_layer = tf.keras.layers.InputLayer(input_shape=(3,))
hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
output_layer = tf.keras.layers.Dense(units=2, activation='softmax')

# 모델을 생성
model = tf.keras.Sequential([
    input_layer,
    hidden_layer,
    output_layer
])

# 모델 컴파일
model.compile(loss='mse', optimizer='Adam')

# 뉴런층속성: 이름
print('뉴런층속성이름')
print(input_layer.name, input_layer.dtype)
print(hidden_layer.name, hidden_layer.dtype)
print(output_layer.name, output_layer.dtype)

# 입력 rank, 출력  rank
print(input_layer.input.shape)
print(input_layer.output.shape)
print(hidden_layer.input.shape)
print(hidden_layer.output.shape)
print(output_layer.input.shape)
print(output_layer.output.shape)

# 활성화 함수이름
print("활성화 함수이름")
print(hidden_layer.activation.__name__)
print(output_layer.activation.__name__)

print("가중치 값 출력")
print(hidden_layer.weights)
print(output_layer.weights) 