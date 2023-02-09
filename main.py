# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave

import matplotlib.pyplot as plt
import pandas as pd
import math

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.ylabel('numbers')
plt.show()


# implementation of growth algorithm

fishMidline = [[0, 2.7, 1.0], [0.1, 3.0, 0.0], [0.2, 2.7, -1.0]]

segmentBeginning = [0, 0]
segmentInitialLength = [3.0, 0]
segmentEnd = segmentInitialLength
segmentIncrement = 0.001

errorThreshold = 0.05

length = 1 # (meters) 

def calculateJoints():
  Joints = []  
  jointIndex = 0
  
  while segmentEnd < 1*length:
    e = calculateSegmentError(segmentEnd, segmentBeginning, fishMidline)
    
    if e < errorThreshold:
      segmentEnd += segmentIncrement
    
    else:
      segmentBeginning = segmentEnd - segmentIncrement
      segmentEnd = segmentBeginning + segmentInitialLength
      
      Joints[jointIndex] = segmentBeginning
      jointIndex += 1
      
  print("All segments built")
  
  return Joints
  


calculateSegmentError(segmentEnd, segmentBeginning, fishMidline):
  
  for i in range (length(fishMidline)):
    a = (segmentEnd[1]-segmentBeginning[1]) * fishMidline[i][1] - (segmentEnd[0]-segmentBeginning[0]) * fishMidline[i][2] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1]
    
    b = math.sqrt((segmentEnd-segmentBeginning)**2 + (segmentEnd - segmentBeginning)**2)
    
  return a/b
  
  
# implementation of equal segments

def createEqualSegments(segmentCount):
  segmentLength = (1*length)/segmentCount
  for i in segments:
    segmentEnd = segmentBeginning + i*segmentLength
    Joints[i] = segmentEnd 
    

# create segments of diminishing size but add up to 1

def createDiminishingSegments(segmentCount, modifier):
  Joints = []
  segmentLength = length 
  for i in range (segmentCount):
    segmentLength =/ 2
    Joints[i] = segmentLength
    
  return Joints
  
  
def plotOGFishdata(fileName):
  filePath = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
  location = filePath + fileName
  
  fileData = pd.read_excel(location) 
  
  
plotOGFishdata("Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls")
  
    