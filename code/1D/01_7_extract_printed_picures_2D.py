import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

number_of_picture = 8

I = mpimg.imread("draw-%d-printed.jpg" % (number_of_picture))
# print(I)
# print(I.shape)

Point_veiws = []
point_for_extension = []

x = 0
for point in I:
    x += 1
    y = 0
    # print(point.shape)
    for p in point:
        y += 1
        z = 0
        for z in p:
            if p[0] > 10 and p[1] > 10 and p[2] > 10:

                coordinate = np.array([y, 500-x])
                point_for_extension.append(coordinate)


plt.imshow(I)
plt.show()

# xxxxxxxx


all_coord = np.array(point_for_extension)
# print(all_coord)

Min_X = min(all_coord[:, 0])
all_coord_list = all_coord[:, 0].tolist()
min_index = all_coord_list.index(Min_X)
# print(min_index)

Coor_Y = all_coord[min_index][-1]
# print(Coor_Y)

new_matrix = []
number = 0
for value in range(all_coord.shape[0]):
    number += 1
    new_X = all_coord[value][0] - Min_X
    new_Y = all_coord[value][1] - Coor_Y
    new_matrix.append([new_X, new_Y])

    STR = " ".join([str(number), "1", str(new_X), str(0), str(new_Y)])
    Point_veiws.append(STR)

new_matrix_array = np.array(new_matrix)
# print(new_matrix_array)


New_coordinate_X = max(new_matrix_array[:, 0]) - 0
New_coordinate_Y = max(new_matrix_array[:, 1]) - min(new_matrix_array[:, 1])
# print(New_coordinate_X)
# print(New_coordinate_Y)

xlo, xhi, ylo, yhi, zlo, zhi = 0, New_coordinate_X, 0, 10, 0, New_coordinate_Y

WrittingIn = open("draw-%d-printed.dat" % (number_of_picture), "w")
WrittingIn.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
              "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (number, 1, xlo, xhi, ylo, yhi, zlo, zhi))

for coordinate in Point_veiws:
    WrittingIn.write(coordinate + "\n")

WrittingIn.close()


X_length = max(new_matrix_array[:, 0]) - min(new_matrix_array[:, 0])

### divide into 100 parts

all_number = 199

All_coordinates_view = []
each_part_length = 6#6.35
extract_coords = []
initial_x = 0  # 这个值是需要观看Point_veiws中最左开始点的‘坐标’
initial_y = 0
extract_coords.append(np.array([initial_x, initial_y]))

arrow = 0
while arrow < all_number:
    if arrow == 0:
        account_coordinates = []
        account_coordinates_values = []
        for coord in new_matrix:
            differ = coord - np.array([initial_x, initial_y])
            Radius = np.sqrt(differ.dot(differ))
            P_diff_value = differ.dot(np.array([1, 0]))
            if np.abs(Radius - each_part_length) < 0.75:
                account_coordinates.append(coord)
                account_coordinates_values.append(P_diff_value)
        # print(account_y)

        max_value = max(account_coordinates_values)
        Index_max = account_coordinates_values.index(max_value)
        # print(Index_max)
        All_coordinates_view.append(account_coordinates)

        X_mean = np.mean(np.array(account_coordinates)[:, 0])
        Y_mean = np.mean(np.array(account_coordinates)[:, 1])

        P = np.array([X_mean, Y_mean])
        T = P - extract_coords[-1]

        extract_coords.append([X_mean, Y_mean])

    else:
        account_coordinates = []
        account_coordinates_values = []
        for coord in new_matrix:
            differ = coord - P
            Radius = np.sqrt(differ.dot(differ))
            P_diff_value = differ.dot(T)
            if np.abs(Radius - each_part_length) < 0.75 and differ.dot(T) > 0:
                account_coordinates.append(coord)
                account_coordinates_values.append(P_diff_value)
        print(arrow)
        # print(account_coordinates_values)
        max_value = max(account_coordinates_values)
        Index_max = account_coordinates_values.index(max_value)
        # print(Index_max)

        All_coordinates_view.append(account_coordinates)

        X_mean = np.mean(np.array(account_coordinates)[:, 0])
        Y_mean = np.mean(np.array(account_coordinates)[:, 1])

        P = np.array([X_mean, Y_mean])
        T = P - extract_coords[-1]

        extract_coords.append([X_mean, Y_mean])

    arrow += 1

# print(extract_coords)
# print(y_extra_coordinate)

extract_array = np.array(extract_coords)
#
True_length = 0
for ii in range(0, all_number):
    # print(ii)
    XY_diff = extract_array[ii + 1] - extract_array[ii]
    deta_z = np.sqrt(XY_diff.dot(XY_diff))
    True_length = True_length + deta_z
#
Extension_ratio = float(True_length / 88.5) # 这个值需要从max-displacement中找

extract_coords = extract_array / Extension_ratio
# print(extract_coords)
x_extra_coordinate = extract_coords[0][0]
y_extra_coordinate = extract_coords[0][-1]
# print(y_extra_coordinate)

extract_coords[:, 0] = extract_coords[:, 0] - x_extra_coordinate
extract_coords[:, 1] = extract_coords[:, 1] - y_extra_coordinate + 0.5
# print(extract_coords)




str_read_pre = []
nub_point = 0
type = 0
for coordinate in All_coordinates_view:
    type += 1
    for po in coordinate:
        nub_point += 1
        STR_coord = ' '.join([str(nub_point), str(type), str(po[0]), str(0), str(po[1])])
        str_read_pre.append(STR_coord)

View_Coordinates_In = open('draw-%d-view-printed.dat' % (number_of_picture), 'w')
View_Coordinates_In.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
              "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (nub_point, 199, xlo, xhi, ylo, yhi, zlo, zhi))

for any in str_read_pre:
    View_Coordinates_In.write(any + '\n')

View_Coordinates_In.close()


Target_WrittingIn = open("draw-target-%d-printed.dat" % (number_of_picture), "w")
Target_WrittingIn.write("LAMMPS\n%d atoms\n%d atom types\n%f %f xlo xhi\n"
              "%f %f ylo yhi\n%f %f zlo zhi\n\nAtoms\n\n" % (200, 1, 0, 10.5, 0, 1, -10, 10))
xx = 0
for coordinate in extract_coords:
    xx += 1
    STR_coord = ' '.join([str(xx), '1', str(coordinate[0]), '0', str(coordinate[1])])
    Target_WrittingIn.write(STR_coord + '\n')

Target_WrittingIn.close()

Ouputfile = open("draw-target-%d-printed.txt" % (number_of_picture), "w")
for coordinate in extract_coords:
    STR_coord = '   '.join([str(coordinate[0]), '0', str(coordinate[1])])
    Ouputfile.write(STR_coord + '\n')

Ouputfile.close()

print('OK, done')


