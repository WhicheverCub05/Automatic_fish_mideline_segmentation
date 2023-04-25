# this file contains demonstrations of algorithms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import main
import glob
import calculate_error as ce
import time

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


# when called by matplotlib FuncAnimation, it shows how area is calculated to find error using area
def animate_area_error(i):
    global data_frame
    global fish_midline
    global ani

    segment_beginning = fish_midline[0][0]
    segment_end = fish_midline[data_frame][0]

    y1 = segment_beginning[1]
    y2 = segment_end[1]

    x1 = segment_beginning[0]
    x2 = segment_end[0]

    print("y1:", y1, " y2:", y2, " x1:", x1, " x2:", x2)
    if x2 - x1 != 0:
        gr = (y2 - y1) / (x2 - x1)
    else:
        gr = 0

    c = y1 - (gr * x1)

    x = np.linspace(fish_midline[0][0][0], fish_midline[data_frame][0][0])
    y = (gr * x) + c
    if abs(gr) > 0:
        plt.fill_between(x, fish_midline[data_frame][0][1], step="pre", alpha=0.4)
    old_line = plt.plot(x, y, '-b')

    error = ce.find_area_error(0, data_frame, 0, fish_midline)
    print("Error:", error, " I:", i, " data_frame:", data_frame)
    plt.scatter(fish_midline[data_frame][0][0], fish_midline[data_frame][0][1], label='%d' % error)
    plt.legend(loc="upper right")
    if data_frame + 10 < len(fish_midline) - 1:
        data_frame += 10
    else:
        print("Done")
        ani.pause()

    error_line = old_line.pop(0)
    error_line.remove()


# when called by matplotlib FuncAnimation, it shows the midline being generated step by step
# Also, when called by FuncAnimation, 'i' is incremented each time
def linear_demonstration_animated(i):
    threshold_error = 3
    max_error = 0
    global fish_midline
    global joint_beginning
    global joints
    plt.cla()

    for j in range(len(joints)):
        plt.plot(joints[j][0], joints[j][1], 'black')

    data = []

    for r in range(0, len(fish_midline), 10):
        data.append(fish_midline[r][0])

    for d in data:
        plt.plot(d[0], d[1], 'bo', ls='--')

    plt.scatter(fish_midline[0][0][0], fish_midline[0][0][1], color='red', label='head')
    plt.legend(loc="upper left")

    i %= len(data)
    if i / len(data)-1 == 1:
        joint_beginning = 0
    end = i

    segment_beginning = data[joint_beginning]
    segment_end = data[end]

    y1 = segment_beginning[1]
    y2 = segment_end[1]

    x1 = segment_beginning[0]
    x2 = segment_end[0]

    if x2 - x1 != 0:
        gr = (y2 - y1) / (x2 - x1)
    else:
        gr = 0

    c = y1 - (gr * x1)

    x = np.linspace(data[joint_beginning][0], data[end][0])
    y = (gr * x) + c
    plt.plot(x, y, 'black')

    # iterate from segment_beginning index to segment_end index, incrementing midline_point index
    for j in range(joint_beginning, end, 1):
        midline_point = data[j]

        error = ce.find_linear_error(segment_beginning, segment_end, midline_point)

        c_intersection = midline_point[1] - ((-1 / gr) * midline_point[0])
        x_intersection = abs((c - c_intersection) / (gr - (-1 / gr)))
        y_intersection = ((-1 / gr) * x_intersection) + c_intersection

        x_per = np.linspace(x_intersection, data[j][0])
        ym = ((-1 / gr) * x_per) + midline_point[1] - ((-1 / gr) * midline_point[0])

        plt.scatter(x_intersection, y_intersection) # label='%.2f' % error

        if max_error < error:
            max_error = error

        if error > threshold_error:
            plt.plot(x_per, ym, '-r', linestyle='--')
        else:
            plt.plot(x_per, ym, '-c', linestyle='--')

        if j >= len(data) - 1:
            print("Done?", "j:", j, "end:", end, "len(data)", len(data))
            time.sleep(5)

    if max_error > threshold_error:
        joint_beginning = end - 1
        i -= 2
        joints.append([fish_midline[joint_beginning][0][0], fish_midline[joint_beginning][0][1]])
        plt.scatter(fish_midline[joint_beginning][0][0]+1, fish_midline[joint_beginning][0][1]+1, color='black',
                    label='joint %d' % joint_beginning)
        plt.legend(loc="upper left")
        print("joint_beginning:", joint_beginning, "(%.2f, %.2f)" % (fish_midline[joint_beginning][0][0],
                                                                     fish_midline[joint_beginning][0][1]))

    if end >= len(data)-1:
        print("Can't do any more")
        plt.cla()
        joint_beginning = 0
        time.sleep(5)
        i = 0


# run code only when called as a script
if __name__ == "__main__":
    data_frame = 0
    joints = []
    directory = main.set_data_folder()
    fish_midline = main.load_midline_data(glob.glob(directory + '/*.xls')[0])  # first Excel file in directory
    main.plot_midline(fish_midline, 0)

    while 1:
        user_option = input("Which error method would you like to see? (l:linear, a:area):").capitalize()

        if user_option == "L":
            joint_beginning = 0
            ani = animation.FuncAnimation(fig, linear_demonstration_animated, interval=500)
            plt.show()
        elif user_option == "A":
            ani = animation.FuncAnimation(fig, animate_area_error, interval=500)
            plt.show()
        else:
            print("Please try again")
