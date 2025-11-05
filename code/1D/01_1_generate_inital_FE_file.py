from abaqus import *
from abaqusConstants import *
from math import *
import time
import numpy as np
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()

jobnunmber = 5001
while jobnunmber <= 8000:

    differ = jobnunmber - 0
    if differ % 5 == 0:
        time.sleep(60)
    else:
        pass

    Mdb()
    # create model
    if mdb.models.has_key("Model-1"):
        myModel = mdb.models["Model-1"]
    else:
        myModel = mdb.Model(name="Model-1", modelType=STANDARD_EXPLICIT)

    myModel.Material(name='yz-expansion')
    myModel.materials['yz-expansion'].setValues(description='')
    myModel.materials['yz-expansion'].Elastic(dependencies=0, moduli=LONG_TERM, noCompression=OFF, noTension=OFF,
                                              table=((18600.0, 0.22),),
                                              temperatureDependency=OFF, type=ISOTROPIC)
    myModel.materials['yz-expansion'].Density(dependencies=0, distributionType=UNIFORM, fieldName='',
                                              table=((3.95e-09,),),
                                              temperatureDependency=OFF)
    myModel.materials['yz-expansion'].Expansion(dependencies=0, table=((-0.16, -0.08, -0.08, 0.0, 0.0, 0.0),),
                                                temperatureDependency=OFF, type=ANISOTROPIC, userSubroutine=OFF,
                                                zero=0.0)
    myModel.materials['yz-expansion'].setValues(materialIdentifier='')

    # myModel.Material(name='xz-expansion')
    # myModel.materials['xz-expansion'].setValues(description='')
    # myModel.materials['xz-expansion'].Elastic(dependencies=0, moduli=LONG_TERM, noCompression=OFF, noTension=OFF,
    #                                           table=((377000.0, 0.24),),
    #                                           temperatureDependency=OFF, type=ISOTROPIC)
    # myModel.materials['xz-expansion'].Density(dependencies=0, distributionType=UNIFORM, fieldName='',
    #                                           table=((3.95e-09,),),
    #                                           temperatureDependency=OFF)
    # myModel.materials['xz-expansion'].Expansion(dependencies=0, table=((0.1, 0.01, 0.1, 0.0, 0.0, 0.0),),
    #                                             temperatureDependency=OFF, type=
    #                                             ANISOTROPIC, userSubroutine=OFF, zero=0.0)
    # myModel.materials['xz-expansion'].setValues(materialIdentifier='')

    myModel.Material(name='xy-expansion')
    mdb.models['Model-1'].materials['xy-expansion'].setValues(description='')
    mdb.models['Model-1'].materials['xy-expansion'].Elastic(dependencies=0, moduli=LONG_TERM, noCompression=OFF,
                                                            noTension=OFF, table=((31800.0, 0.22),),
                                                            temperatureDependency=OFF, type=ISOTROPIC)
    mdb.models['Model-1'].materials['xy-expansion'].Density(dependencies=0, distributionType=UNIFORM, fieldName='',
                                                            table=((3.95e-09,),),
                                                            temperatureDependency=OFF)
    mdb.models['Model-1'].materials['xy-expansion'].Expansion(dependencies=0, table=((-0.08, -0.08, -0.16, 0.0, 0.0, 0.0),),
                                                              temperatureDependency=OFF, type=
                                                              ANISOTROPIC, userSubroutine=OFF, zero=0.0)
    mdb.models['Model-1'].materials['xy-expansion'].setValues(materialIdentifier='')

    myModel.HomogeneousSolidSection(material='yz-expansion', name='Section-1', thickness=None)
    # myModel.HomogeneousSolidSection(material='xz-expansion', name='Section-2', thickness=None)
    myModel.HomogeneousSolidSection(material='xy-expansion', name='Section-2', thickness=None)

    myAssembly = myModel.rootAssembly
    myAssembly.DatumCsysByDefault(CARTESIAN)

    width = 5
    height = 0.5
    length = 2.5

    # create base
    numerx = 40
    numery = 1

    countnum = 0
    z = 0
    while z < 2:
        zz = height * z
        y = 0
        while y < numery:
            yy = length * y
            x = 0
            while x < numerx:
                xx = length * x

                countnum += 1

                myPart = myModel.Part(name="Part-%d" % (countnum), dimensionality=THREE_D, type=DEFORMABLE_BODY)
                mySketch = myModel.ConstrainedSketch(name="sketch-1", sheetSize=200)
                mySketch.rectangle(point1=(xx, yy), point2=(xx + length, yy + width))
                myPart.BaseSolidExtrude(sketch=mySketch, depth=height)

                cells = myPart.cells
                region = regionToolset.Region(cells=cells)

                randomnumber = np.random.randint(1, 3)
                myPart.SectionAssignment(region=region, sectionName='Section-%d' % (randomnumber), offset=0.0,
                                         offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

                myPart.Set(cells=myPart.cells, name='Set-%d-%d-%d-%d' % (countnum, x, y, z))

                myAssembly.Instance(name="Part-%d" % (countnum), part=myPart, dependent=ON)
                myAssembly.translate(instanceList=("Part-%d" % (countnum),), vector=(0, 0, zz))

                x += 1
            y += 1
        z += 1

    instances = []
    for ins in myAssembly.instances.values():
        instances.append(ins)
    myAssembly.InstanceFromBooleanMerge(name='Part-all', instances=tuple(instances), keepIntersections=ON,
                                        originalInstances=SUPPRESS, domain=GEOMETRY)

    cells = myModel.parts['Part-all'].cells#.getSequenceFromMask(mask=('[#3ffff ]',), )
    region = regionToolset.Region(cells=cells)

    myModel.parts['Part-all'].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_1, fieldName='',
                                                  localCsys=None, orientationType=GLOBAL,
                                                  region=region, stackDirection=STACK_3)

    myModel.parts['Part-all'].Set(faces=myModel.parts['Part-all'].faces.getSequenceFromMask((
        '[#0:4 #10000000 #0:6 #10 ]',), ), name='Set-left-side')

    myModel.EncastreBC(createStepName='Initial', localCsys=None,
                                     name='BC-1', region=myAssembly.instances['Part-all-1'].sets['Set-left-side'])

    myModel.parts['Part-all'].seedPart(size=0.5, deviationFactor=0.5, minSizeFactor=0.5)
    myModel.parts['Part-all'].generateMesh()

    myModel.StaticStep(name='Step-1', previous='Initial', nlgeom=ON)
    # myModel.steps['Step-1'].setValues(initialInc=0.01)
    myModel.fieldOutputRequests['F-Output-1'].setValues(timeInterval=1.0)

    c1 = myAssembly.instances['Part-all-1'].cells
    f1 = myAssembly.instances['Part-all-1'].faces
    e1 = myAssembly.instances['Part-all-1'].edges
    v1 = myAssembly.instances['Part-all-1'].vertices
    region = regionToolset.Region(vertices=v1, edges=e1, faces=f1, cells=c1)
    myModel.Temperature(name='Predefined Field-1',
                        createStepName='Initial', region=region, distributionType=UNIFORM,
                        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(100.0,))

    myModel.Temperature(name='Predefined Field-2',
                        createStepName='Step-1', region=region, distributionType=UNIFORM,
                        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(101.0,))

    mdb.Job(name='Job-%d' % (jobnunmber), model='Model-1', description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,
            numGPUs=0)
    mdb.jobs['Job-%d' % (jobnunmber)].submit(consistencyChecking=OFF)

    jobnunmber += 1

