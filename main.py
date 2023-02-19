# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave
import math
import matplotlib.pyplot as plt
import pandas as pd

# implementation of growth algorithm

error_threshold = 0.22


# get data from every other line for now
def load_midline_data(location):
    file_data = pd.read_excel(location)
    dimensions = file_data.shape
    print("shape: ", dimensions)
    
    midline = [[[0 for _ in range(2)] for _ in range(
        int(dimensions[1] / 2))] for _ in range(dimensions[0])]

    for column in range(0, dimensions[1], 2):
        for row in range(dimensions[0]):
            x = file_data.iat[row, column]
            y = file_data.iat[row, column + 1]
            midline[row][int(column / 2)][0] = x
            midline[row][int(column / 2)][1] = y

    return midline


def plot_midline(midline, *columns):

    if columns:
        for c in columns:
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][c][0])
                y.append(midline[s][c][1])
            plt.plot(x, y)

    else:
        for c in range(len(midline[0])):
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][c][0])
                y.append(midline[s][c][1])
            plt.plot(x, y)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def generate_segments(midline):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([0.0, 0.0, 0])  # contains x, y, and increment

    frames = len(midline[0])
    print("Len midline: ", frames)

    segment_beginning = [0, 0]
    segment_end = [0, 0]
    increments = 0

    while increments < len(midline):  # number of rows
        error = 0

        for f in range(3):  # len(midline[0]) all columns

            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]

            segment_end[0] = midline[increments][f][0]
            segment_end[1] = midline[increments][f][1]

            a = abs((segment_end[1] - segment_beginning[1]) * midline[increments][f][0]
                    - (segment_end[0] - segment_beginning[0])
                    * midline[increments][f][1] + segment_beginning[0] * segment_end[1] - segment_end[0] *
                    segment_beginning[1])
            b = math.sqrt((segment_end[0] - segment_beginning[0])
                          ** 2 + (segment_end[1] - segment_beginning[1]) ** 2)

            if a != 0.0 or b != 0.0:
                error += (a / b) / 100

        error /= 3
        # error /= len(midline[0])

        if error < error_threshold:
            increments += 1
            print("ye: ", increments, " error: ", error)

        elif error >= error_threshold:
            increments -= 1

            if increments <= joints[len(joints) - 1][2]:
                print("stuck on increment: ", increments)
                break
            else:
                joints.append([midline[increments][0][0],
                               midline[increments][0][1], increments])
                print("Adding joint: ", joints[len(joints) - 1])

        else:
            print("Houston, we have a problem")

    return joints


# implementation of equally divided segments
def create_equal_segments(segment_count, midline, frame):
    joints = [[0 for _ in range(3)] for _ in range(0)]

    for i in segment_count:
        increment = int((i / segment_count) * len(midline))
        x = midline[increment][frame][0]
        y = midline[increment][frame][1]
        joints.append([x, y, increment])
    return joints


# create segments of diminishing size but add up to 1
def create_diminishing_segments(segment_count, length):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    segment_length = length
    for i in range(segment_count):
        segment_length /= 2
        joints.append(segment_length)

    return joints


def main():
    # file_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    file_path = r"C:\Users\Which\Desktop\uni\Y3\Main_assignment\Data\Sturgeon from Elsa and Ted\midlines/"
    file_name = "Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls"

    fish_midline = load_midline_data(file_path + file_name)

    joints = generate_segments(fish_midline)

    print("==========================")

    print("size of joints = ", len(joints), ", Joints: ", joints)

    for j in range(len(joints)):
        plt.scatter(joints[j][0], joints[j][1], color='green')

    plot_midline(fish_midline, 0)

    plt.show()


main()
