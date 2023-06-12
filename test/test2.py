# 머신 러닝 기초에 대한 ㅁ누제를 풀어서 제출하세요

# 문제2 입력값이 1개 출력값이 1개인 간단한 신경망을 구현하고
# 손실함수 = 'mean_squared_error'
# 옵티마이즈 = 'sgd'를 사용해서 컴파일시키는 test2.py 코드를 작성하시오 (5줄 이하로 구현)

# 답안 2
import tensorflow as tf
test2 = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
test2.compile(loss='mean_squared_error', optimizer='sgd')
