# Automatic_fish_mideline_segmentation
This program aims to generate fish segments based on midline points which describe the way that the fish is curved using different techniques. 
It can also be used to fit straight lines to data points.

Data must be saved in the .xls format and graphs are saved as .svg files to retain their quality.
Each .xls file in that data folder produces one graph using all the data points within that .xls file.

data format in a table is:

                   data set. each: [x, y]
data_points: |   x   |   y   |   x   |   y   | ...
             |   x   |   y   |   x   |   y   | 
                            ...

A midline must have 2 columns each, which has a x and y value (int or float) on each column. 
Each row describes a single point in the data. 
There is no known limit to number of columns or rows.


Modules used:
- math,
- matplotlib
- pandas
- numpy
- os
- sys
- glob
- csv

Arguments that can be passed:
location of fish midline data, location to save graphical data


