import keras
import pandas as pd
import numpy as np
import os
workpath = os.getcwd()
files = '%s' % (workpath)

arrow = 0
while arrow < 10:

    random_pick_numb = np.random.randint(1, 1000)
    origin_file = pd.read_csv('inpout-file-%d.txt' % (random_pick_numb), sep='  ', names=['0', '1', '2', '3', '4', '5', '6'],
                              engine='python')
    Data = np.array(origin_file)
    Data_Sorted = sorted(Data, key=lambda x: (x[3], x[2]))
    Data = np.array(Data_Sorted)



    XData = np.array(Data[:, 0:2])

    ### test the mean and std
    str_mean_std = np.array([np.mean(XData), np.std(XData)])
    XData_New = (XData - 2) / 0.81
    # print(XData_New)

    ### for the coordinates maps
    coord_x_y = np.array(Data[:, 2:4])
    # print(coord_x_y)
    normal_coord_xy = (coord_x_y - np.mean(coord_x_y)) / np.std(coord_x_y)
    Data_input = np.concatenate((XData_New, normal_coord_xy), axis=1)
    # print(Data_input)
    XIndata = np.reshape(Data_input, (20, 20, 4))

    alltraininput = []
    alltraininput.append(XIndata)

    X_Coorinate_Data = Data[:, -3]
    Y_Coorinate_Data = Data[:, -2]
    Z_Coorinate_Data = Data[:, -1]

    CNN_model_y = keras.models.load_model("CNN-model-y.keras")
    CNN_predict_y_values = CNN_model_y.predict(np.array(alltraininput))

    CNN_model_x = keras.models.load_model("CNN-model-x.keras")
    CNN_predict_x_values = CNN_model_x.predict(np.array(alltraininput))
    # print(CNN_predict_x_values)
    # print(CNN_predict_x_values.shape)
    CNN_model_z = keras.models.load_model("CNN-model-z.keras")
    CNN_predict_z_values = CNN_model_z.predict(np.array(alltraininput))

    Compare_file = open('Compare-sample-%d.dat' % (random_pick_numb), 'w')
    Compare_file.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
                       "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (
                           800, 2, 0, 10, 0, 1, -5, 5))

    length = len(X_Coorinate_Data)
    for i in range(0, length):
        newinput = ' '.join(
            [str(i + 1), str(1), str(float(X_Coorinate_Data[i])), str(float(Y_Coorinate_Data[i])), str(float(Z_Coorinate_Data[i]))])
        Compare_file.write(newinput + '\n')

    for i in range(0, length):
        newinput = ' '.join(
            [str(i + 401), str(2), str(CNN_predict_x_values[0][i]), str(CNN_predict_y_values[0][i]), str(CNN_predict_z_values[0][i])])
        Compare_file.write(newinput + '\n')

    Compare_file.close()


    arrow += 1

