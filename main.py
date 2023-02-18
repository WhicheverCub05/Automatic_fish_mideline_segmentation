# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave
import math
import matplotlib.pyplot as plt
import pandas as pd

# implementation of growth algorithm

errorThreshold = 0.1

# get data from every other line for now
def loadMidlineData(fileName):
    filePath = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    location = filePath + fileName

    fileData = pd.read_excel(location)

    dimentions = fileData.shape

    print("shape: ", dimentions)

    midline = [[[0 for d in range(2)] for x in range(
        int(dimentions[1]/2))] for y in range(dimentions[0])]

    for column in range(0, dimentions[1], 2):
        for row in range(dimentions[0]):
            x = fileData.iat[row, column]
            y = fileData.iat[row, column+1]
            midline[row][int(column/2)][0] = x
            midline[row][int(column/2)][1] = y

    return midline


def plotOGFishdata(midline):

    for f in range(len(midline[0])):
        x = []
        y = []
        for s in range(200):
            # plt.plot(midline[0][0][i])
            x.append(midline[s][f][0])
            y.append(midline[s][f][1])
        plt.plot(x, y)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def generateSegments(length, midline):
    joints = [[0 for x in range(3)] for y in range(0)]
    joints.append([0.0, 0.0, 0])    # contains x, y, and increment

    frames = len(midline[0])
    print("Len midline: ", frames)

    segmentBeginning = [0, 0]
    segmentEnd = [0, 0]
    increments = 0
    error = 0

    while increments < len(midline):    # number of rows
        error = 0

        for f in range(len(midline[0])):    # number of columns

            segmentBeginning[0] = midline[joints[len(joints)-1][2]][f][0]
            segmentBeginning[1] = midline[joints[len(joints)-1][2]][f][1]

            segmentEnd[0] = midline[increments][f][0]
            segmentEnd[1] = midline[increments][f][1]

            a = abs((segmentEnd[1]-segmentBeginning[1]) * midline[increments][f][0] - (segmentEnd[0]-segmentBeginning[0])
                    * midline[increments][f][1] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1])
            b = math.sqrt((segmentEnd[0]-segmentBeginning[0])
                            ** 2 + (segmentEnd[1] - segmentBeginning[1])**2)

            if (a != 0.0 or b != 0.0):
                error += (a/b)/100

        error /= len(midline[0])

        if error < errorThreshold:
            increments += 1

        elif error >= errorThreshold:
            increments -= 1

            if increments <= joints[len(joints)-1][2]:
                print("stuck on increment: ", increments)
                break
            else:
                joints.append([midline[increments][0][0],
                                midline[increments][0][1], increments])
                print("Adding joint: ", joints[len(joints)-1])

            print("Error: ", error)

        else:
            print("Houston, we have a problem")

    return joints


# implementation of equally divided segments
def createEqualSegments(segmentCount, midline, frame):
    joints = [[0 for x in range(3)] for y in range(0)]

    for i in segmentCount:
        increment = int((i/segmentCount)*len(midline))
        x = midline[increment][frame][0]
        y = midline[increment][frame][1]
        joints.append([x, y, increment])
    return joints


# create segments of diminishing size but add up to 1
def createDiminishingSegments(segmentCount, length):
    joints = [[0 for x in range(3)] for y in range(0)]
    segmentLength = length
    for i in range(segmentCount):
        segmentLength /= 2
        joints.append(segmentLength)

    return joints


def main():
    fishMidline = loadMidlineData(
        "Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls")

    length = 0.102

    joints = generateSegments(length, fishMidline)

    print("==========================")

    print("size of joints = ", len(joints), ", Joints: ", joints)

    for j in range(len(joints)):
        plt.plot(joints[j])

    plt.show()

    plt.show()


main()
