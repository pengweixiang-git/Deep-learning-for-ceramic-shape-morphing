import numpy as np
import pandas as pd
import keras

number_of_picture = 3

origin_file = open('best-solution-draw-target-%d.txt' % (number_of_picture))
all_list_file = origin_file.readlines()
Data = []
for xx in all_list_file:
    list_xx = xx.split()
    Data.append([int(list_xx[0]), int(list_xx[1])])


Origin_xy = pd.read_csv('inpout-file-1.txt', sep='  ',
                        names=['0', '1', '2', '3', '4', '5', '6'], engine='python')
Coodinate = np.array(Origin_xy)
# print(Data)
coord_x_y = np.array(Coodinate[:, 2:4])
# print(coord_x_y)
normal_coord_xy = (coord_x_y - np.mean(coord_x_y)) / np.std(coord_x_y)


N = 1
full_input = []

N_Individual = []
for chromosome in Data:
    N_Individual.extend([chromosome] * 16)  # 复制16次

New_individual = np.array(N_Individual)  # (200, 2)
XData_Normal = (New_individual - 2) / 0.81
Data_input = np.concatenate((XData_Normal, normal_coord_xy), axis=1)

New_individual_coord_sort = sorted(Data_input, key=lambda x: (x[3], x[2]))
# print(np.array(New_individual_coord_sort))

Inpdata = np.reshape(New_individual_coord_sort, (20, 20, 4))

full_input.append(Inpdata)
full_input = np.array(full_input)  # shape: (N, 1600, 3, 1)

CNN_model_x = keras.models.load_model("CNN-model-x.keras")
CNN_model_y = keras.models.load_model("CNN-model-y.keras")
CNN_model_z = keras.models.load_model("CNN-model-z.keras")

CNN_predict_x_values = np.reshape(CNN_model_x.predict(full_input, verbose=0), (1, 400, 1))  # (N, 400, 1)
# print(CNN_predict_x_values.shape)
CNN_predict_y_values = np.reshape(CNN_model_y.predict(full_input, verbose=0), (1, 400, 1))  # (N, 400, 1)
CNN_predict_z_values = np.reshape(CNN_model_z.predict(full_input, verbose=0), (1, 400, 1))  # (N, 400, 1)

predict_x_y = np.concatenate([CNN_predict_x_values, CNN_predict_y_values, CNN_predict_z_values], axis=2)
print(predict_x_y)

Target_WrittingIn = open("Compare-draw-predict-target-%d.dat" % (number_of_picture), "w")
Target_WrittingIn.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
              "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (1200, 3, 0, 55, 0, 1, 0, 50))
number_atoms = 0

Open_draw_txt = pd.read_csv('draw-target-output-file-%d.txt' % (number_of_picture), sep=' ', names=['0', '1', '2'], engine='python')
# print(Open_draw_txt)
Draw_Data = np.array(Open_draw_txt)
for coord in Draw_Data:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '1', str(coord[0]), str(coord[1]), str(coord[2])])
    Target_WrittingIn.write(STR_coord + '\n')

for coordinate in predict_x_y[0]:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '2', str(coordinate[0]), str(coordinate[1]), str(coordinate[2])])
    Target_WrittingIn.write(STR_coord + '\n')



FEM_draw_txt = pd.read_csv('inpout-file-draw-%s.txt' % (number_of_picture), sep='  ', names=['0', '1', '2', '3', '4', '5', '6'], engine='python')
FEM_Draw_Data = np.array(FEM_draw_txt)
Data_Sorted = sorted(FEM_Draw_Data, key=lambda x: (x[3], x[2]))
Data = np.array(Data_Sorted)
# print(Data)

X_Y_Data = np.array(Data[:, [-3, -2, -1]])

for coord in X_Y_Data:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '3', str(coord[0]), str(coord[1]), str(coord[2])])
    Target_WrittingIn.write(STR_coord + '\n')


Target_WrittingIn.close()