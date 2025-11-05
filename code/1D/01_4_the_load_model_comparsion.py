import keras
import pandas as pd
import numpy as np
import os
workpath = os.getcwd()
files = '%s' % (workpath)

arrow = 0
while arrow < 5:

    random_pick_numb = np.random.randint(5000, 7000)
    origin_file = pd.read_csv('inpout-file-%d.txt' % (random_pick_numb), sep='  ', names=['0', '1', '2', '3', '4', '5'],
                              engine='python')
    Data = np.array(origin_file)
    Data_Sorted = sorted(Data, key=lambda x: (x[2]))
    Data = np.array(Data_Sorted)

    XData = np.array(Data[:, 0:2])

    XData_Normal = (XData - np.mean(XData)) / np.std(XData)

    coord_x_y = np.reshape(np.array(Data[:, 2]), (200, 1))
    # print(coord_x_y)
    normal_coord_xy = (coord_x_y - np.mean(coord_x_y)) / np.std(coord_x_y)
    # print(np.mean(coord_x_y))
    # print(np.std(coord_x_y))

    Data_input = np.concatenate((XData_Normal, normal_coord_xy), axis=1)

    Inpdata = np.reshape(Data_input, (200, 3, 1))
    alltraininput = []
    alltraininput.append(Inpdata)

    X_Coorinate_Data = Data[:, 3]
    Y_Coorinate_Data = Data[:, -1]

    CNN_model_y = keras.models.load_model("CNN-model-y-8000.keras")
    CNN_predict_y_values = CNN_model_y.predict(np.array(alltraininput)) #* 25.298446278596455 + 0.657714217344769

    CNN_model_x = keras.models.load_model("CNN-model-x-8000.keras")
    CNN_predict_x_values = CNN_model_x.predict(np.array(alltraininput)) #* 21.029829582793187 + 27.96297128977711
    # print(CNN_predict_x_values)
    # print(CNN_predict_x_values.shape)

    Compare_file = open('Compare-sample-%d.dat' % (random_pick_numb), 'w')
    Compare_file.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
                       "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (
                           400, 2, 0, 10, 0, 1, -5, 5))

    length = len(X_Coorinate_Data)
    for i in range(0, length):
        newinput = ' '.join(
            [str(i + 1), str(1), str(float(X_Coorinate_Data[i])), str(0), str(float(Y_Coorinate_Data[i]))])
        Compare_file.write(newinput + '\n')

    for i in range(0, length):
        newinput = ' '.join(
            [str(i + 201), str(2), str(CNN_predict_x_values[0][i]), str(0), str(CNN_predict_y_values[0][i])])
        Compare_file.write(newinput + '\n')

    Compare_file.close()


    arrow += 1

