from tensorflow import keras

mse_model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])

#mse 설정
mse_model.compile(loss='mse') #mean_squared_error 동일-약어
# optimizer = 'rmsprop'

xt = [0]
pred = mse_model.predict(xt)
print(pred)
# 0, 1, 0 실제값

mse_model.evaluate(xt, [[0,1,0]]) 