import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.shuffle(5)
dataset = dataset.repeat(2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()

for i in range():
  value = sess.run(next_element)
  print(value)