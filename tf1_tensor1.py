import tensorflow as tf

scalar = tf.constant(1)
vector = tf.constant([1,2,3])
matrix = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
tensor = tf.constant([[[1,2,3], [4,5,6], [7,8,9]],
                      [[1,2,3], [4,5,6], [7,8,9]],
                      [[1,2,3], [4,5,6], [7,8,9]]])

print(tf.rank(scalar))
print(tf.rank(vector))
print(tf.rank(matrix))
print(tf.rank(tensor))