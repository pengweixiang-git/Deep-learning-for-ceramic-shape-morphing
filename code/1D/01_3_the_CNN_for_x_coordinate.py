import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


alltraininput = []
alltrainoutput = []
alltestinput = []
alltestoutput = []
allXpredict = []
allYpredict = []


direction = 'x'

#### input is x, turn on the activation function, for y, turn off the activation

if direction == 'x':
    slice_number = 3
else:
    slice_number = -1


all_y_values = []

countnumb = 1
arrow = 1
while arrow <= countnumb:
    # with open('inpout-file-%d.txt' % (arrow), "rb") as f:
    #     print(f.read(2))
    Open_files = open('inpout-file-%d.txt' % (arrow), 'r')
    Read_datas = Open_files.readlines()
    # print(Read_datas)
    Orig_data = []
    for ele_line in Read_datas:
        ele_line_split = ele_line.split()
        line_to_numb = [int(ele_line_split[0]), int(ele_line_split[1]), float(ele_line_split[2]), float(ele_line_split[3]), float(ele_line_split[4]), float(ele_line_split[5])]
        # print(line_to_numb)
        Orig_data.append(line_to_numb)

    Data = np.array(Orig_data)
    # print(Data)
    Data_Sorted = sorted(Data, key=lambda x: (x[2]))
    Data = np.array(Data_Sorted)
    print(Data)

    XData = np.array(Data[:, 0:2])

    XData_Normal = (XData - np.mean(XData)) / np.std(XData)

    coord_x_y = np.reshape(np.array(Data[:, 2]), (200,1))
    # print(coord_x_y)
    normal_coord_xy = (coord_x_y - np.mean(coord_x_y)) / np.std(coord_x_y)
    # print(np.mean(coord_x_y))
    # print(np.std(coord_x_y))

    Data_input = np.concatenate((XData_Normal, normal_coord_xy), axis=1)

    XIndata = np.reshape(Data_input, (200, 3, 1))
    # print(XIndata)

    YData = Data[:, slice_number]

    # all_y_values.append(YData)

    YDataNew = YData # - 27.96297128977711) / 21.029829582793187

    # print(YDataNew)
    if arrow <= countnumb*0.99:
        alltraininput.append(XIndata)
        alltrainoutput.append(YDataNew)
    else:
        alltestinput.append(XIndata)
        alltestoutput.append(YDataNew)

    # if arrow == countnumb + 1:
    #     allXpredict.append(XIndata)
    #     allYpredict.append(YDataNew)
    #     os.system('cp inpout-file-%d.txt inpout-file-predict.txt' % (countnumb + 1))

    Open_files.close()

    arrow += 1

AllXtrainData = np.array(alltraininput)
AllYtrainData = np.array(alltrainoutput)
AllXtestData = np.array(alltestinput)
AllYtestData = np.array(alltestoutput)

# print(AllYtrainData)
# print(AllYtrainData.shape)

# reshape_y = np.reshape(all_y_values, (200*countnumb, 1))
# print(np.mean(reshape_y))
# print(np.std(reshape_y))


#
class CNNMnlist(object):

    model = keras.models.Sequential([
        # 卷积层1： 32 个5*5的filter， strides = 1, padding='same'
        keras.layers.Conv2D(64, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # 池化，2*2， step=2
        keras.layers.AvgPool2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        # 卷积2,64个5*5*32的filter，strides=2
        keras.layers.Conv2D(128, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # # 池化层2， 2*2窗口，strids=2, [None, 8*8*64]
        keras.layers.AvgPool2D(pool_size=(2, 1), strides=(2, 1), padding='same'),
        # [None, 15, 15, 64] ---> [None, 15*15*64],全链接层
        keras.layers.Flatten(),
        # # 6400个神经元网络层
        keras.layers.Dense(3200, activation=tf.nn.relu),
        # keras.layers.Dense(1600),#, activation=tf.nn.relu),
        # keras.layers.Dense(800),
        # # 200个神经元
        keras.layers.Dense(200)#, activation=tf.nn.relu)

    ])

    def __init__(self):
        self.x_train, self.y_train = AllXtrainData, AllYtrainData
        self.x_test, self.y_test = AllXtestData, AllYtestData
        # self.x_predict = AllXprediction
        # print(self.x_train, self.y_train)

    def compile(self, epochs=500):
        CNNMnlist.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                                loss=keras.losses.MeanSquaredError(),
                                metrics=['accuracy'])

        def lr_schedule(epoch):
            if epoch < epochs*0.5:
                learn_rate = 0.0001
            elif epoch < epochs*0.75 and epoch >= epochs*0.5:
                learn_rate = 0.0001
            else:
                learn_rate = 0.0001
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
        test_loss, test_acc = CNNMnlist.model.evaluate(self.x_test, self.y_test, batch_size=16)
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
    cnn.model.save('CNN-model-%s-%d.keras' % (direction, countnumb))
    print(cnn.model.summary())

    y_pred = cnn.model.predict(AllXtrainData)
    # print(y_pred)
    # print(AllYtestData)

    R2 = 1 - mean_squared_error(AllYtrainData, y_pred) / np.var(AllYtrainData)
    print(R2)

    # all_data_to_plot = []
    # for i in range(0, len(AllYtrainData)):
    #     predict_value = y_pred[i]
    #     true_value = AllYtrainData[i]
    #     for inside_i in range(0, len(predict_value)):
    #         pre_x = predict_value[inside_i]
    #         tru_y = true_value[inside_i]
    #         str_get = ' '.join([str(pre_x), str(tru_y)])
    #         all_data_to_plot.append(str_get)
    #
    #
    # Output_file_plt = open('a-out-%splt-pre-tru.txt' % (direction), 'w')
    # for ele in all_data_to_plot:
    #     Output_file_plt.write(ele + '\n')
    #
    # Output_file_plt.close()

