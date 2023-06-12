# 머신 러닝 기초에 대한 문제를 풀어서 제출하세요
# 문제4
# 문제 3에서 옵티마이저를 Adam과 RMSprop을 사용하는 경우 각각의 손실값을 나타내는 test4.py 코드를 작성하시오

# 답안4
import tensorflow as tf
# adam
adam = tf.keras.Sequential([tf.keras.layers.Dense(units=3, input_shape=[1])])
adam.compile(loss='mean_squared_error', optimizer='adam')
xs = [1]
ys = [[0,1,0]]
adam.fit(xs, ys, epochs=5, verbose=0)

# rms
rms = tf.keras.Sequential([tf.keras.layers.Dense(units=3, input_shape=[1])])
rms.compile(loss='mean_squared_error', optimizer='rmsprop')
xs = [1]
ys = [[0,1,0]]
rms.fit(xs, ys, epochs=5, verbose=0)
# 출력
print(f'adam 손실값 : {adam.evaluate(xs, ys)}, rms 손실값 : {rms.evaluate(xs, ys)}')