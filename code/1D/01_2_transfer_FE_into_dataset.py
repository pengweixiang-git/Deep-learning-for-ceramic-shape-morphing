# -* - coding:UTF-8 -*-
import sys
import os
import numpy as np
from odbAccess import *
from abaqus import *
from abaqusConstants import *

##
# 这个是将every unit cell的所有点都提取出来
##


workpath = os.getcwd()
files = '%s' % (workpath)

Max_Displacement_all_units = []

arrow = 1
while arrow <= 1:

    OpenInputfile = open('Job-%d.inp' % (arrow), 'r')
    Allinput = OpenInputfile.readlines()

    # after deformed
    odb = openOdb('%s/Job-%d.odb' % (files, arrow))

    step1 = odb.steps['Step-1']
    displacement_last = step1.frames[-1].fieldOutputs['U']

    InOutputarray = []
    i = 1
    while i <= 40:

        eachinput = []
        eachoutput = []

        linenodestr1 = '*Nset, nset=_PickedSet2_#%d, internal' % (i)
        lineelestr1 = '*Elset, elset=_PickedSet2_#%d, internal, generate' % (i)
        linenodestr2 = '*Nset, nset=_PickedSet2_#%d, internal' % (i + 40)
        lineelestr2 = '*Elset, elset=_PickedSet2_#%d, internal, generate' % (i + 40)

        linenodenumb1 = Allinput.index(linenodestr1 + '\n')
        # print(linenodenumb1)
        lineelenumb1 = Allinput.index(lineelestr1 + '\n')
        # print(lineelenumb1)
        downnodes1 = Allinput[linenodenumb1 + 1:lineelenumb1]
        # print(downnodes1)
        setnode1number = []
        for nodenumb in downnodes1:
            eachline = nodenumb.split(',')
            for nodein in eachline:
                # print(nodein)
                setnode1number.append(int(nodein))
        # print(setnode1number)

        linenodenumb2 = Allinput.index(linenodestr2 + '\n')
        # print(linenodenumb2)
        lineelenumb2 = Allinput.index(lineelestr2 + '\n')
        # print(lineelenumb2)
        downnodes2 = Allinput[linenodenumb2 + 1:lineelenumb2]
        # print(downnodes2)
        setnode2number = []
        for nodenumb in downnodes2:
            eachline = nodenumb.split(',')
            for nodein in eachline:
                # print(nodein)
                setnode2number.append(int(nodein))
        # print(setnode2number)

        common_elements = set(setnode1number) & set(setnode2number)
        result_list = list(common_elements)
        # print(result_list)

        materialproperty = []

        materialline1 = '*Solid Section, elset=_PickedSet2_#%d, orientation=Ori-1' % (i)
        materialline2 = '*Solid Section, elset=_PickedSet2_#%d, orientation=Ori-1' % (i+40)

        for material in Allinput:
            if material.startswith(materialline1):
                allmaterialline1 = material
            elif material.startswith(materialline2):
                allmaterialline2 = material

        bottommaterial1linenumb = Allinput.index(allmaterialline1) - 1
        # print(material1linenumb)
        property1 = int(Allinput[bottommaterial1linenumb][-2])
        # print(property1)
        # materialproperty.append(int(property1))
        # print(materialproperty)

        uppermaterial2linenumb = Allinput.index(allmaterialline2) - 1
        # print(material2linenumb)
        property2 = int(Allinput[uppermaterial2linenumb][-2])
        # print(property2)


        material_array = [property1 - 1, property2 - 1]

        # ## 判断material的组合并将其转化成常数，仅仅在RNN中使用
        # if material_array[0] == 1:
        #     if material_array[1] == 1:
        #         if material_array[2] == 1:
        #             if material_array[3] == 1:
        #                 material_numb = 0
        #             else:
        #                 material_numb = 1
        #         else:
        #             if material_array[3] == 1:
        #                 material_numb = 2
        #             else:
        #                 material_numb = 3
        #     else:
        #         if material_array[2] == 1:
        #             if material_array[3] == 1:
        #                 material_numb = 4
        #             else:
        #                 material_numb = 5
        #         else:
        #             if material_array[3] == 1:
        #                 material_numb = 6
        #             else:
        #                 material_numb = 7
        #
        # else:
        #     if material_array[1] == 1:
        #         if material_array[2] == 1:
        #             if material_array[3] == 1:
        #                 material_numb = 8
        #             else:
        #                 material_numb = 9
        #         else:
        #             if material_array[3] == 1:
        #                 material_numb = 10
        #             else:
        #                 material_numb = 11
        #     else:
        #         if material_array[2] == 1:
        #             if material_array[3] == 1:
        #                 material_numb = 12
        #             else:
        #                 material_numb = 13
        #         else:
        #             if material_array[3] == 1:
        #                 material_numb = 14
        #             else:
        #                 material_numb = 15

        materialproperty.append(material_array)
        # print(materialproperty)

        allnodesget = []
        dispalcement = []
        allnodesupper = Allinput.index('*Node' + '\n')
        for nodenumb in result_list:
            nodeinfo = Allinput[allnodesupper + nodenumb]
            # print(nodeinfo)
            nodeinfolist = nodeinfo.split(',')
            point = np.array([float(nodeinfolist[1]), float(nodeinfolist[2]), float(nodeinfolist[3])])
            displacementValues_last = displacement_last.values[nodenumb - 1].magnitude
            dispalcement.append(displacementValues_last)
            allnodesget.append(point)
        # nodesgetarray = np.array(allnodesget)
        Max_XY_Vaules = [np.max(np.array(allnodesget)[:, 0]), np.max(np.array(allnodesget)[:, 1])]
        Max_displacement = np.max(dispalcement)
        Max_Displacement_all_units.append(Max_displacement)
        # print(dispalcement)
        # print(Max_XY_Vaules)

        for nodenumb in result_list:
            nodeinfo = Allinput[allnodesupper + nodenumb]
            # print(nodeinfo)
            nodeinfolist = nodeinfo.split(',')
            point = np.array([float(nodeinfolist[1]), float(nodeinfolist[2]), float(nodeinfolist[3])])
            # print(point)
            displacementValues_last = displacement_last.values[nodenumb - 1].data
            # print(displacementValues_last)
            if point[0] < Max_XY_Vaules[0] and point[1] < 0.000001: # point[1]限制Y方向的维度，此处1维的线
                Deformed_coordinate = point + displacementValues_last
                InOutputarray.append([material_array, point, Deformed_coordinate])

        # for array in allnodesget:
        #     eachinput.append([np.array(array), np.array(materialproperty)])

        # after deformation
        # for numb in result_list:
        #     displacementValues_last = displacement_last.values[numb-1].data
        #     # translist = [displacementValues_last[0], displacementValues_last[1], displacementValues_last[2]]
        #     # # print(translist)
        #     # transarray = np.array(translist)
        #     # print(transarray)
        #     eachoutput.append(displacementValues_last)
        #
        # print(eachoutput)

        i += 1

    OpenInputfile.close()
    odb.close

    OpenInOuputfile = open('inpout-file-%d.txt' % (arrow), 'w')
    for xx in InOutputarray:
        # print(xx[0][0])
        allstr = '  '.join([str(xx[0][0]), str(xx[0][1]), str(str(xx[1][0])), str(str(xx[2][0])), str(str(xx[2][1])), str(str(xx[2][2]))])
        OpenInOuputfile.write(allstr + '\n')

    OpenInOuputfile.close()

    arrow += 1


print('OK, done')



