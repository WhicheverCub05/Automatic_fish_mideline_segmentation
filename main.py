# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave

import matplotlib.pyplot as plt 
import math

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.ylabel('numbers')
plt.show()


# implementation of growth algorithm

midline = [[0, 2.7, 1.0], [0.1, 3.0, 0.0], [0.2, 2.7, -1.0]]

segmentBeginning = [0, 0]
segmentInitialLength = [3.0, 0]
segmentEnd = segmentInitialLength
segmentIncrement = 0.001

errorThreshold = 0.05

def calculateJoints():
  Joints = []  
  jointIndex = 0
  
  while segmentEnd < 1:
    e = calculateSegmentError(segmentEnd, segmentBeginning, midline)
    
    if e < errorThreshold:
      segmentEnd += segmentIncrement
    
    else:
      segmentBeginning = segmentEnd - segmentIncrement
      segmentEnd = segmentBeginning + segmentInitialLength
      
      Joints[jointIndex] = segmentBeginning
      jointIndex += 1
      
  print("All segments built")
  
  return Joints
  


calculateSegmentError(segmentEnd, segmentBeginning, midline):
  
  for i in range (length(midline)):
    a = (segmentEnd[1]-segmentBeginning[1]) * midline[i][1] - (segmentEnd[0]-segmentBeginning[0]) * midline[i][2] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1]
    
    b = math.sqrt((segmentEnd-segmentBeginning)**2 + (segmentEnd - segmentBeginning)**2)
    
  return a/b
  
  
# implementation of equal segments

def createEqualSegments(segmentCount):
  segmentLength = 1/segmentCount
  for i in segments:
    segmentEnd = segmentBeginning + i*segmentLength
    Joints[i] = segmentEnd 
    

# create segments of diminishing size but add up to 1

def createDiminishingSegments(segmentCount):
  
  
    