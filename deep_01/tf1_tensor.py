import tensorflow as tf

scalar = tf.constant(1)
vector = tf.constant([1,2,3])
matrix = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
tensor = tf.constant([[[1,2,3], [4,5,6], [7,8,9]],
                      [[1,2,3], [4,5,6], [7,8,9]],
                      [[1,2,3], [4,5,6], [7,8,9]]])

# print(tf.rank(scalar))
# print(tf.rank(vector))
# print(tf.rank(matrix))
# print(tf.rank(tensor))


a_1 = tf.zeros(5)
print(a_1)
a_11 = tf.zeros([2])
print(a_11)
a_2 = tf.zeros([2,2])
print(a_2)
a_3 = tf.zeros([3,3,3])
print(a_3)

# 텐서의 형태 확인
a_vector = tf.constant([2,3])
print(a_vector.shape)
b_matrix = tf.constant([ [2,2], [3,3] ])
print(b_matrix.shape)

# 텐서를 넘파이로 변환
a_vector1 = tf.constant([1,2,3,4,5])
print(a_vector1)
a_np1 = a_vector1.numpy()
print(a_np1)

# 넘파이를 텐서로 변환
import numpy as np
np_array = np.array([[1,2,3],[4,5,6]])
print(np_array)
tf_matrix = tf.convert_to_tensor(np_array)
print(tf_matrix) 