# 머신 러닝 기초에 대한 문제를 풀어서 제출하세요
# 문제3
# 입력값이 1개이고 출력값이 3개인 간단한 신경망을 구성하고
# 손실함수 = 'mean_squared_error'
# 옵티마이즈 = 'sgd'를 사용하여 훈련시킬 때
# 입력값[0], 출력값 [[0,1,0]] 에 대한 손실값을 출력하는 test3.py 코드를 작성하시오
# 답안 3

import tensorflow as tf
test3 = tf.keras.Sequential([tf.keras.layers.Dense(units=3, input_shape=[1])])
test3.compile(loss='mean_squared_error', optimizer='sgd')
xs = [1]
ys = [[0,1,0]]
loss_value = test3.fit(xs, ys, epochs=5 ,verbose=0)
print('손실값 :', test3.evaluate(xs, ys))