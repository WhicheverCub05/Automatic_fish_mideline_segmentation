# this file contains demonstrations of algorithms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import main
import glob
import calculate_error as ce

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate_area_error(i):
    global data_frame
    global area_error_midline
    global ani

    segment_beginning = area_error_midline[0][0]
    segment_end = area_error_midline[data_frame][0]

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

    x = np.linspace(area_error_midline[0][0][0], area_error_midline[data_frame][0][0])
    y = (gr * x) + c
    if abs(gr) > 0:
        plt.fill_between(x, area_error_midline[data_frame][0][1], step="pre", alpha=0.4)
    old_line = plt.plot(x, y, '-b')

    error = ce.find_area_error(0, data_frame, 0, area_error_midline)
    print("Error:", error, " I:", i, " data_frame:", data_frame)
    plt.scatter(area_error_midline[data_frame][0][0], area_error_midline[data_frame][0][1], label='%d' % error)
    plt.legend(loc="upper right")
    if data_frame + 10 < len(area_error_midline) - 1:
        data_frame += 10
    else:
        print("Done")
        ani.pause()

    error_line = old_line.pop(0)
    error_line.remove()


def linear_demonstration_animated(i):
    max_error = 0
    plt.cla()
    data = [[0.7, 1.0], [1.7, 1.5], [2.3, 2.7], [2.8, 3.7], [3.7, 4], [5, 4], [6.3, 2]]

    for d in data:
        plt.plot(d[0], d[1], 'bo', ls='--')

    start = 0
    end = i % len(data)

    segment_beginning = data[start]
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

    x = np.linspace(data[start][0], data[end][0])
    print("type: ", type(x))
    y = (gr * x) + c
    plt.plot(x, y, 'black')

    # iterate from segment_beginning index to segment_end index, incrementing midline_point index
    for i in range(start + 1, end, 1):
        midline_point = data[i]

        error = ce.find_linear_error(segment_end, segment_beginning, midline_point)
        
        c_intersection = midline_point[1] - ((-1 / gr) * midline_point[0])
        x_intersection = abs((c - c_intersection) / (gr - (-1 / gr)))
        y_intersection = ((-1 / gr) * x_intersection) + c_intersection

        x_per = np.linspace(x_intersection, data[i][0])
        ym = ((-1 / gr) * x_per) + midline_point[1] - ((-1 / gr) * midline_point[0])

        plt.scatter(x_intersection, y_intersection, label='(%.1f, %.1f),%.2f' % (x_intersection, y_intersection, error))
        plt.annotate('(%.1f, %.1f),%.2f' % (x_intersection, y_intersection, error),
                     [x_intersection + 0.15, y_intersection + 0.15])

        if max_error > error:
            max_error = error
            plt.plot(x_per, ym, '-r', linestyle='dashed')
        else:
            plt.plot(x_per, ym, '-c', linestyle='dashed')

    plt.legend(loc="upper right")
    plt.ylim(0, 5)
    plt.xlim(0, 6.5)


# run code only when called as a script
if __name__ == "__main__":
    data_frame = 0
    directory = main.set_data_folder()
    area_error_midline = main.load_midline_data(glob.glob(directory + '/*.xls')[0])  # first Excel file in directory

    while 1:
        user_option = input("Which error method would you like to see? (l:linear, a:area):").capitalize()

        if user_option == "L":
            ani = animation.FuncAnimation(fig, linear_demonstration_animated, interval=2000, frames=10)
            plt.show()
        elif user_option == "A":
            main.plot_midline(area_error_midline, 0)
            ani = animation.FuncAnimation(fig, animate_area_error, interval=400, frames=10)
            plt.show()
        else:
            print("Please try again")
