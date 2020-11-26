import tensorflow as tf

# Here I set it default to relu so that our variable became of that type
act_func_global = tf.nn.relu
print(act_func_global)
is_Swish = False
print('is a Swish function: ', is_Swish)
is_SwishBeta = False
print('is a Swish Beta function: ', is_SwishBeta)
is_First = True
print('is a first function: ', is_Swish)
b_size=500
print('batch size: ', b_size)
