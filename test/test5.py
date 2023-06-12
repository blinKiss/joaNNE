# 문제5 그림과 같은 신경망을 구현하는 코드를 작성하는 test5.py코드를 작성하시오
# 단 은닉층에서는 relu를 출력층에서는 softmax를 활성화 함수로 사용하시오

# 답안5
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3,)),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(loss='mse', optimizer='Adam')
