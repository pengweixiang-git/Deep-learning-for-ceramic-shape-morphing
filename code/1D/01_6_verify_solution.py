import numpy as np
import pandas as pd
import keras

number_of_picture = 8

origin_file = pd.read_csv('best-solution-draw-target-%d.txt' % (number_of_picture), sep='  ', names=['0', '1'], engine='python')
Data = np.array(origin_file)
# print(Data)

# new_data = []
# for xx in Data:
#     list_xx = xx[0].split('\t')
#     # print(list_xx)
#     new_data.append([int(list_xx[0]),
#                      int(list_xx[1]),
#                      int(list_xx[2]),
#                      int(list_xx[3])])
#
# print(new_data)
# Data = np.array(new_data)



New_individual = []
for chromosome in Data:
    for _ in range(5):
        New_individual.append(chromosome)

# print(New_individual)

coord = np.array([i / 2 for i in range(200)]).reshape(200, 1)
normal_coord_xy = (coord - np.mean(coord)) / np.std(coord)
XData_Normal = (New_individual - np.mean(New_individual)) / np.std(New_individual)
Data_input = np.concatenate((XData_Normal, normal_coord_xy), axis=1)

Inpdata = np.reshape(Data_input, (200, 3, 1))
alltraininput = []
alltraininput.append(Inpdata)

CNN_model_y = keras.models.load_model("CNN-model-y-8000.keras")
CNN_model_x = keras.models.load_model("CNN-model-x-8000.keras")

CNN_predict_x_values = CNN_model_x.predict(np.array(alltraininput))
# print(CNN_predict_x_values)

CNN_predict_y_values = CNN_model_y.predict(np.array(alltraininput))
# print(CNN_predict_y_values)


predict_x_y = np.concatenate((np.reshape(CNN_predict_x_values, (200, 1)), np.reshape(CNN_predict_y_values, (200, 1))),
                             axis=1)

Target_WrittingIn = open("Compare-draw-predict-target-%d.dat" % (number_of_picture), "w")
Target_WrittingIn.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
              "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (600, 3, 0, 55, 0, 1, 0, 50))
number_atoms = 0

Open_draw_txt = pd.read_csv('draw-target-%d.txt' % (number_of_picture), sep='  ', names=['0', '1', '2'], engine='python')
# print(Open_draw_txt)
Draw_Data = np.array(Open_draw_txt)
for coord in Draw_Data:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '1', str(coord[0]), str(coord[1]), str(coord[2])])
    Target_WrittingIn.write(STR_coord + '\n')

for coordinate in predict_x_y:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '2', str(coordinate[0]), '0', str(coordinate[1])])
    Target_WrittingIn.write(STR_coord + '\n')



FEM_draw_txt = pd.read_csv('inpout-file-draw-%d.txt' % (number_of_picture), sep='  ', names=['0', '1', '2', '3', '4', '5'], engine='python')
FEM_Draw_Data = np.array(FEM_draw_txt)
X_Y_Data = np.array(FEM_Draw_Data[:, [3, -1]])
# print(X_Y_Data)

for coord in X_Y_Data:
    number_atoms += 1
    STR_coord = ' '.join([str(number_atoms), '3', str(coord[0]), '0', str(coord[1])])
    Target_WrittingIn.write(STR_coord + '\n')


Target_WrittingIn.close()