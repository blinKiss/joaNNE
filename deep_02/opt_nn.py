from tensorflow import keras

opt_model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

# 손실, 최적화함수 적용
opt_model.compile(loss='mse', optimizer='sgd')

# 훈련데이터
xs = [1]
ys = [[0,1,0]] # 결과

loss_his = opt_model.fit(xs, ys, epochs=100)
# xt = [1]
# result = opt_model.predict(xt)
# print(result)
opt_model.evaluate([1], [[0,1,0]])

# loss_his 값을 그래프로 그리기
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['font.size'] = 14

loss = loss_his.history['loss']
plt.plot(loss)
plt.xlabel('count')
plt.ylabel('loss')
plt.show()