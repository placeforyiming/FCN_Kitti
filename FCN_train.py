from FCN import FCN
import tensorflow as tf

'''
tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.00001,epoch=25,train_continue=20)


tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.000004,epoch=30,train_continue=25)


tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.000001,epoch=35,train_continue=30)


tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.0000005,epoch=40,train_continue=35)

'''

tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.0000002,epoch=45,train_continue=40)



tf.reset_default_graph()
A=FCN()
A.build(type='FCN_8')
A.train_object(batch_size=2,learning_rate=0.0000001,epoch=50,train_continue=45)
