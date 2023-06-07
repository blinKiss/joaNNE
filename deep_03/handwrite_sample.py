import tensorflow as tf

mnist = tf.keras.datasets.mnist
# 2차원 데이터
(x_train, y_train), (x_test, y_test)  = mnist.load_data()
# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0

# 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 훈련
model.fit(x_train, y_train, epochs=5)

# 손실과 정확도 출력
loss, accuracy = model.evaluate(x_test, y_test)
print('훈련테스트 손실도:', loss)
print('훈련테스트 정확도:', accuracy)

loss_data = []
accuracy_data = []
for i in range(10):
    model.fit(x_train, y_train, epochs=1)
    loss_data.append(model.evaluate(x_test, y_test)[0])
    accuracy_data.append(model.evaluate(x_test, y_test)[1])

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4,3)
plt.rcParams['font.size'] = 12

plt.plot(loss_data) # y
plt.ylim([0.1, 0.0])
plt.xlabel('count')
plt.ylabel('loss')
plt.show()
