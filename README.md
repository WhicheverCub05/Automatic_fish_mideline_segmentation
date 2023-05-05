# Automatic_fish_mideline_segmentation

Author: Alexandre Roque da Silva
Date: 05/05/2023

This program uses different techniques to generate joints for robotic fish using real fish data.

The program can also be used to approximate any series of waves to a single set of points along the series' data which area the joints.

Running a generation method for a midline produces a single graph per midline file in the .svg format.

Midline data must be saved in the .xls format. 

data format in a table is:

                   data set. each:  |frame n|frame  | ...
                                    | x | y | x | y |
                                    | x | y | x | y |
                                    | x | y | x | y |
                                     ...

A midline must have 2 columns each, which has a x and y value (int or float) on each column. 
Each row describes a single point in the data. 
There is no known limit to number of columns or rows.


Dependencies used:
- math
- matplotlib
- pandas
- numpy
- glob
- csv

Arguments that can be passed:
location of fish midline data, location to save graphical data

The program is divided into 6 files:
- calculate_error.py : 
  - contains methods to calculate error between joint and midline
  

- gather_data.py :
  - used to gather data


- demonstration.py :
  - used to demonstrate the growth method


- generation_methods_linear_error.py:
  - contains generation methods that use linear error 


- generation_methods_area_error.py:
  - contains generation methods that use area error 

- main.py:
  - contains the functions load data, and CLI