import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

workpath = os.getcwd()
files = '%s' % (workpath)
# print(files)

alltraininput = []
alltrainoutput = []
alltestinput = []
alltestoutput = []
allXpredict = []
allYpredict = []


direction = 'y'

#### input is x, turn on the activation function, for y, turn off the activation

if direction == 'x':
    slice_number = -3
elif direction == 'y':
    slice_number = -2
else:
    slice_number = -1

output_mean_std = []

countnumb = 31158
arrow = 1
while arrow <= countnumb:
    Originadata = pd.read_csv('%s/inpout-file-%d.txt' % (files, arrow), sep='  ', names=['0', '1', '2', '3', '4', '5', '6'], engine='python')
    # print(Originadata.shape)
    Data = np.array(Originadata)
    # print(Data)
    Data_Sorted = sorted(Data, key=lambda x: (x[3], x[2]))
    Data = np.array(Data_Sorted)
    # print(Data)

    XData = np.array(Data[:, 0:2])

    ### test the mean and std
    str_mean_std = np.array([np.mean(XData), np.std(XData)])
    output_mean_std.append(str_mean_std)
    XData_New = (XData - 2) / 0.81
    # print(XData_New)

    ### for the coordinates maps
    coord_x_y = np.array(Data[:, 2:4])
    # print(coord_x_y)
    normal_coord_xy = (coord_x_y - np.mean(coord_x_y)) / np.std(coord_x_y)
    Data_input = np.concatenate((XData_New, normal_coord_xy), axis=1)
    # print(Data_input)
    XIndata = np.reshape(Data_input, (20, 20, 4))
    # print(XIndata)

    YData = Data[:, slice_number]
    # print(YData)
    # print(YData.shape)
    # YDataNew = (YData - np.mean(YData)) / np.std(YData)

    YDataNew = YData
    # print(YData[0])
    # print(YDataNew)
    if arrow <= countnumb*0.9:
        alltraininput.append(XIndata)
        alltrainoutput.append(YDataNew)
    else:
        alltestinput.append(XIndata)
        alltestoutput.append(YDataNew)

    arrow += 1

AllXtrainData = np.array(alltraininput)
AllYtrainData = np.array(alltrainoutput)
AllXtestData = np.array(alltestinput)
AllYtestData = np.array(alltestoutput)

# all_mean = np.max(np.array(output_mean_std)[:, 0])
# all_std = np.mean(np.array(output_mean_std)[:, 1])
# print(all_mean)
# print(all_std)



class CNNMnlist(object):

    model = keras.models.Sequential([
        # 卷积层1： 32 个5*5的filter， strides = 1, padding='same'
        keras.layers.Conv2D(64, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # 池化，2*2， step=2
        keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        # 卷积2,64个5*5*32的filter，strides=2
        keras.layers.Conv2D(128, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # # 池化层2， 2*2窗口，strids=2, [None, 8*8*64]
        keras.layers.AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        # [None, 15, 15, 64] ---> [None, 15*15*64],全链接层
        keras.layers.Flatten(),
        # # 6400个神经元网络层
        keras.layers.Dense(1600, activation=tf.nn.relu),
        # keras.layers.Dense(1600),#, activation=tf.nn.relu),
        # keras.layers.Dense(800),
        # # 200个神经元
        keras.layers.Dense(400)#, activation=tf.nn.relu)

    ])

    def __init__(self):
        self.x_train, self.y_train = AllXtrainData, AllYtrainData
        self.x_test, self.y_test = AllXtestData, AllYtestData
        # self.x_predict = AllXprediction
        # print(self.x_train, self.y_train)

    def compile(self, epochs=500):
        CNNMnlist.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                loss=keras.losses.MeanSquaredError(),
                                metrics=['accuracy'])

        def lr_schedule(epoch):
            if epoch < epochs*0.5:
                learn_rate = 0.001
            elif epoch < epochs*0.75 and epoch >= epochs*0.5:
                learn_rate = 0.001 # * 0.5 * 0.5 * 0.5
            else:
                learn_rate = 0.001 #* 0.5 * 0.5 * 0.5 * 0.5 * 0.5
            return learn_rate

        # 创建学习率调度器
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

        # 开始训练，使用学习率调度器
        CNNMnlist.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=16, callbacks=[lr_scheduler],
                            validation_data=(self.x_test, self.y_test))

    # 迭代次数运行样品的次数， 整个数据集被分的大小
    # def fit(self):
    #     CNNMnlist.model.fit(self.x_train, self.y_train, epochs=100, batch_size=8)
    #     return None

    def evaluate(self):
        test_loss, test_acc = CNNMnlist.model.evaluate(self.x_test, self.y_test)
        print(test_loss, test_acc)

    # def prediction(self):
    #     predictedvalue = CNNMnlist.model.predict(self.x_predict)
    #
    #     Ouput_predict_file = open('predict-file-%s.txt' % (direction), 'w')
    #
    #     for coordinate in predictedvalue[0]:
    #         Ouput_predict_file.write(str(coordinate) + '\n')
    #
    #     Ouput_predict_file.close()



if __name__ == '__main__':
    cnn = CNNMnlist()
    cnn.compile()
    # cnn.fit()
    cnn.evaluate()
    # cnn.prediction()
    cnn.model.save('CNN-model-%s.keras' % (direction))
    print(cnn.model.summary())

    y_pred = cnn.model.predict(AllXtrainData)
    # print(y_pred)
    # print(AllYtestData)

    R2 = 1 - mean_squared_error(AllYtrainData, y_pred) / np.var(AllYtrainData)
    print(R2)



