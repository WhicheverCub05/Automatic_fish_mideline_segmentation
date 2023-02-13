# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave

import matplotlib.pyplot as plt
import pandas as pd
import math



# implementation of growth algorithm

fishMidline = [[0, 2.7, 1.0], [0.1, 3.0, 0.0], [0.2, 2.7, -1.0]]

segmentBeginning = [0, 0]
segmentInitialLength = [0, 0]
segmentEnd = segmentInitialLength


errorThreshold = 0.05


#length in meters
def calculateJoints(length, frames):
  
  joints = []
  jointIndex = 0
  segmentIncrement = length/200 # a 200th of the fish length = their 

  while segmentEnd < length:
    error = calculateSegmentError(segmentBeginning, segmentEnd, fishMidline, frames)
    
    if error < errorThreshold:
      segmentEnd += segmentIncrement
    
    else:
      segmentBeginning = segmentEnd - segmentIncrement
      segmentEnd = segmentBeginning + segmentInitialLength
      
      joints[jointIndex] = segmentBeginning
      jointIndex += 1
      
  print("All segments built")

  return joints
  


def calculateSegmentError(segmentEnd, segmentBeginning, fishMidline, frames):
  
  for i in range (frames):
    a = (segmentEnd[1]-segmentBeginning[1]) * fishMidline[i][1] - (segmentEnd[0]-segmentBeginning[0]) * fishMidline[i][2] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1]
    
    b = math.sqrt((segmentEnd-segmentBeginning)**2 + (segmentEnd - segmentBeginning)**2)
    
  return a/b
  
  
# implementation of equal segments

def createEqualSegments(segmentCount, length):
  joints = []
  segmentLength = (length)/segmentCount
  for i in segmentCount:
    segmentEnd = segmentBeginning + i*segmentLength
    joints[i] = segmentEnd 
  return joints
    

# create segments of diminishing size but add up to 1

def createDiminishingSegments(segmentCount, length, modifier):
  joints = []
  segmentLength = length 
  for i in range (segmentCount):
    segmentLength /= 2
    joints[i] = segmentLength
    
  return joints
  
  
def plotOGFishdata(fileName):
  filePath = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
  location = filePath + fileName

  fileData = pd.read_excel(location)

  #x_axis = fileData[]
  #y_axis = fileData[]

  #plt.bar(x_axis, y_axis, width=7)

  plt.plot(fileData)

  #plt.plot(fileData)

  #fileData.head()

  plt.xlabel("Segment")
  plt.ylabel("Position")

  plt.show()
  

plotOGFishdata("Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls")
  
"""plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.ylabel('numbers')
plt.show()"""