from VGG16.Vgg16 import Vgg16
import tensorflow as tf
import numpy as np
from math import ceil
import os
import random
import cv2
import matplotlib.pyplot as plt

import scipy.misc



class FCN:
    def __init__(self):

        self.N_class=20
        self.checkpoint_path='./save/'
        self.type=""

    def build(self,type="FCN_8"):
        self.type=type+'_'
        self.rgb_scaled = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        vgg = Vgg16()
        (self.Pool3, self.Pool4, self.Pool5) = vgg.build(self.rgb_scaled)
        self.temp = set(tf.all_variables())

        if type=="FCN_8":
            self.conv_6 = self.conv_layer(self.Pool5, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name= "conv_6")
            self.conv_7 = self.conv_layer(self.conv_6, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name="conv_7")

            self.Score_5 = self.Score_to_class(self.conv_7, "Pool5_score",filter=self.Pool4.get_shape()[3])
            self.Score_5_plus = self.upsample(self.Score_5,tf.shape(self.Pool4), self.Pool4.get_shape()[3] , '5_plus', ksize=4, stride=2)


            self.Score_4 = self.Score_to_class(self.Pool4+self.Score_5_plus, "Pool4_score",filter=self.Pool3.get_shape()[3])
            self.Score_4_plus = self.upsample(self.Score_4,tf.shape(self.Pool3), self.Pool3.get_shape()[3], '4_plus', ksize=4, stride=2)

            self.Score_3 = self.Score_to_class(self.Pool3+self.Score_4_plus, "Pool3_score")
            self.Output = self.upsample(self.Score_3,tf.shape(self.rgb_scaled), self.N_class, 'Output', ksize=16, stride=8)

        if type == "FCN_16":
            self.conv_6 = self.conv_layer(self.Pool5, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name="conv_6")
            self.conv_7 = self.conv_layer(self.conv_6, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name="conv_7")

            self.Score_5 = self.Score_to_class(self.conv_7, "Pool5_score", filter=self.Pool4.get_shape()[3])

            self.Score_5_plus = self.upsample(self.Score_5,tf.shape(self.Pool4), self.N_class, '5_plus', ksize=4, stride=2)

            self.Score_4 = self.Score_to_class(self.Pool4, "Pool4_score")
            self.Score_4_plus_5=tf.add(self.Score_4,self.Score_5_plus)

            self.Output = self.upsample(self.Score_4_plus_5,tf.shape(self.rgb_scaled), self.N_class, 'Output', ksize=32, stride=16)

        if type == "FCN_32":
            self.conv_6 = self.conv_layer(self.Pool5, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name="conv_6")
            self.conv_7 = self.conv_layer(self.conv_6, filter=512, kernel=[3, 3], stride=1, padding='SAME',layer_name="conv_7")

            self.Score_5 = self.Score_to_class(self.conv_7, "Pool5_score", filter=self.Pool4.get_shape()[3])

            self.Output=self.upsample(self.Score_5, tf.shape(self.rgb_scaled),self.N_class, 'Output', ksize=64, stride=32)
        self.output = tf.sigmoid(self.Output)

        return self.Output

    def Dataset(self,train=1):
        dataset=[]
        if train==1:
            Img_number_dir = os.listdir('./Dataset/original_image/train/')
            for i in range(len(Img_number_dir)):
                a = plt.imread('./Dataset/original_image/train/' + Img_number_dir[i])
                b = np.load('./Dataset/label_image/train/' + Img_number_dir[i][:(len(Img_number_dir[i]) - 4)]+'.npy')
                dataset.append({'image':a*255,'label':b})

        else:
            Img_number_dir = os.listdir('./Dataset/original_image/test/')
            for i in range(len(Img_number_dir)):
                a = plt.imread('./Dataset/original_image/test/' + Img_number_dir[i])
                b=Img_number_dir[i]
                dataset.append({'image': a*255,'link':b})
        return dataset

    def train_object(self, batch_size=8, learning_rate=0.00001, epoch=15, train_continue=0):
        # This function is going to train the nn for output the binary value of each object
        # it will train epoch-train_continue epochs from train_continue epoch

        self.data = self.Dataset(train=1)

        print(len(self.data))

        self.label_image = tf.placeholder(dtype=tf.float32, shape=(None, None, None, self.N_class))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_image, logits=self.Output)

        lost = tf.reduce_mean(cross_entropy)

        Learning_rate = tf.placeholder(tf.float32, None)

        trainer = tf.train.AdamOptimizer(Learning_rate)

        gvs = trainer.compute_gradients(lost)

        def ClipGradient(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        clip_gradient = []
        for grad, var in gvs:
            clip_gradient.append((ClipGradient(grad), var))
        train_step = trainer.apply_gradients(clip_gradient)
        with tf.Session() as sess:
            # initialize uninitialized variable
            sess.run(tf.initialize_variables(set(tf.all_variables()) - self.temp))
            if epoch <= train_continue:
                print("You can't go back")
            if train_continue > 0:
                saver = tf.train.Saver()
                saver.restore(sess, self.checkpoint_path + self.type + str(train_continue) + '.ckpt')
            else:
                saver = tf.train.Saver()

            Index_train = [i for i in range(len(self.data))]
            for e in range(train_continue, epoch):
                worse_image = []
                random.shuffle(Index_train)
                start = 0
                Total_accuracy = 0.0
                while ((start + batch_size) < len(Index_train)):
                    Image_batch = []
                    Image_label = []

                    for i in range(batch_size):

                        Image_batch.append(self.data[start]['image'][:, :, :3])
                        label_raw = self.data[start]['label']
                        label_matrix = np.zeros(shape=(np.shape(label_raw)[0], np.shape(label_raw)[1], self.N_class))
                        for m in range(np.shape(label_raw)[0]):
                            for n in range(np.shape(label_raw)[1]):
                                label_matrix[m, n, int(label_raw[m, n, 0])] = 1

                        Image_label.append(label_matrix)
                        start = start + 1
                    train_step.run(feed_dict={self.rgb_scaled: Image_batch, self.label_image: Image_label,
                                              Learning_rate: learning_rate})
                    Output = self.Output.eval(feed_dict={self.rgb_scaled: Image_batch})
                    accuracy = 0.0

                    for i in range(batch_size):
                        accr_per = np.mean(np.argmax(Output[i, :, :, :], axis=2) == np.argmax(Image_label[i], axis=2))
                        worse_image.append(accr_per)
                        accuracy = accuracy + accr_per

                    Total_accuracy = Total_accuracy + accuracy
                    if start % 500 == 0:
                        print(Total_accuracy / start)
                print("The average accuracy for epoch %d is :" % e)
                print(Total_accuracy / start)

                if e > 0:
                    for r in range(1):
                        print(len(worse_image))
                        quater = np.percentile(worse_image, 30)
                        count = 0
                        Image_label = []
                        Image_batch = []
                        for i in range(len(worse_image)):
                            Label = self.data[i]['label']
                            if worse_image[i] < quater and np.mean(np.int32(Label[:, :, 0]) == 19) < 0.05:
                                count = count + 1
                                Image_batch.append(self.data[i]['image'][:, :, :3])
                                label_raw = self.data[i]['label']
                                label_matrix = np.zeros(
                                    shape=(np.shape(label_raw)[0], np.shape(label_raw)[1], self.N_class))
                                for m in range(np.shape(label_raw)[0]):
                                    for n in range(np.shape(label_raw)[1]):
                                        label_matrix[m, n, int(label_raw[m, n, 0])] = 1
                                Image_label.append(label_matrix)
                                if len(Image_batch) == batch_size:
                                    train_step.run(
                                        feed_dict={self.rgb_scaled: Image_batch, self.label_image: Image_label,
                                                   Learning_rate: learning_rate})
                                    Image_batch = []
                                    Image_label = []
                        print(count)

            saver = tf.train.Saver()
            saver.save(sess, self.checkpoint_path + self.type + str(epoch) + '.ckpt')

    def inference_object(self,epoch,im_show=50,istrain=0,refine=False):
        self.data = self.Dataset(train=istrain)
        Pixel = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

        Index_test = [i for i in range(len(self.data))]
        print (len(Index_test))
        with tf.Session() as sess:
            #initialize uninitialized variable
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint_path+self.type+str(epoch)+'.ckpt')
            start = 0
            while (start  < len(Index_test)):
                print (start)
                Image_batch = []

                Image_batch.append(self.data[start]['image'][:, :, :3])
                Output = self.Output.eval(feed_dict={self.rgb_scaled: Image_batch})
                result_matrix = Output[0, :, :, :]
                result_image = np.zeros(shape=(np.shape(result_matrix)[0], np.shape(result_matrix)[1], 3))

                if refine:
                    output_image=np.zeros(shape=(np.shape(result_matrix)[0], np.shape(result_matrix)[1]))
                    for m in range(np.shape(result_matrix)[0]):
                        for n in range(np.shape(result_matrix)[1]):
                            result_class = np.argmax(result_matrix[m, n, :19])

                            output_image[m,n]=result_class
                    refine_image=self.Corse_img(output_image)

                for m in range(np.shape(result_image)[0]):
                    for n in range(np.shape(result_image)[1]):
                        if refine:
                            #print (result_class)
                            result_class=int(refine_image[m,n])
                        else:
                            result_class=np.argmax(result_matrix[m,n,:19])

                        if result_class == 19:
                            color = (0, 0, 0)
                        else:
                            color = Pixel[result_class]
                        result_image[m, n, 0] = color[0] / 255.0
                        result_image[m, n, 1] = color[1] / 255.0
                        result_image[m, n, 2] = color[2] / 255.0

                if start<im_show:
                    plt.imshow(self.data[start]['image'][:, :, :3]/255.0)
                    plt.show()
                    plt.imshow(result_image)
                    plt.show()

                scipy.misc.toimage(np.uint8(result_image * 255)).save('./Output/'+ self.data[start]['link'])

                start = start + 1


    def Corse_img(self,img,rec_range=3):
        def refine_center(long_mat):
            n_length = np.shape(long_mat)[0]
            new_mat = long_mat[int((n_length - 1) / 2), :]
            for i in range(np.shape(long_mat)[1]):
                if i >= (n_length - 1) / 2 and (i + (n_length - 1) / 2) < np.shape(long_mat)[1]:
                    recep_range = long_mat[:, (i - rec_range):(i + rec_range + 1)]
                    recep_range = np.asarray(recep_range).reshape(-1)
                    recep_range = np.int64(recep_range)
                    new_mat[i] = np.argmax(np.bincount(recep_range))
            return new_mat
        Shape=np.shape(img)
        result_image=img
        count=0
        for m in range(Shape[0]):
            if m>=rec_range and (m+rec_range)<Shape[0] :
                count=count+1
                result_image[m]=refine_center(img[(m-rec_range):(m+rec_range+1),:])
        assert count+rec_range*2==Shape[0]
        return result_image

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,shape=weights.shape)
        return var

    def upsample(self, input, shape, out_channel, name,ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = input.get_shape()[3].value
            new_shape=[shape[0],shape[1],shape[2],out_channel]
            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, out_channel, in_features]

            # create
            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(input, weights, output_shape,
                                            strides=strides, padding='SAME')
        return deconv

    def Score_to_class(self,input,name,filter=0):
        with tf.variable_scope(name):
            if filter==0:
                self.score_1 = self.conv_layer(input, filter=128, kernel=[1, 1], stride=1,padding='SAME', layer_name=name+"score_1")
                self.score_2 = self.conv_layer(self.score_1, filter=self.N_class, kernel=[1, 1], stride=1,padding='SAME', layer_name=name+"score_2")
            else:
                self.score_2 = self.conv_layer(input, filter=filter, kernel=[1, 1], stride=1,padding='SAME', layer_name=name + "score_2")

            return self.score_2

    def conv_layer(self,input, filter, kernel, stride=1, padding='SAME', layer_name="conv", act=None):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride,
                                       padding=padding, activation=act)
            return network
