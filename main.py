# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave

import matplotlib.pyplot as plt
import pandas as pd
import math

# implementation of growth algorithm

errorThreshold = 0.1 # some reason 2.48 changes for col 0


# get data from every other line for now
def loadMidlineData(fileName):
  filePath = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
  location = filePath + fileName
  
  fileData = pd.read_excel(location)

  dimentions = fileData.shape

  print("shape: ", dimentions)

  midline = [[[0 for d in range(2)] for x in range(int(dimentions[1]/2))] for y in range(dimentions[0])]

  for column in range(0, dimentions[1], 2):
    for row in range(dimentions[0]):
      x = fileData.iat[row, column]
      y = fileData.iat[row, column+1]
      midline[row][int(column/2)][0] = x
      midline[row][int(column/2)][1] = y
  
  for f in range (len(midline[0])):
    x = []
    y = []
    for s in range(200):
      #plt.plot(midline[0][0][i])
      x.append(midline[s][f][0])
      y.append(midline[s][f][1])
    plt.plot(x, y)
  #plt.show()

  return midline


def plotOGFishdata(fileName):
  filePath = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
  location = filePath + fileName

  fileData = pd.read_excel(location)

  plt.plot(fileData)

  plt.xlabel("Segment")
  plt.ylabel("Position")

  #plt.show()
  

#length in meters
def createJointsGeneration(length, midline):
  
  joints = [[0 for x in range(2)] for y in range(0)]
  
  segmentIncrement = length/200 # a 200th of the fish length
  segmentBeginning = [0, 0]
  increments = 0

  #segmentBeginning[1] = midline.iat[0, 0]
  #segmentEnd = segmentBeginning.copy()

  plt.scatter(increments, segmentEnd[1], color="green") #ls='solid'

  while segmentEnd[0] < length and increments < 199:

    error = calculateSegmentError(segmentBeginning, segmentEnd, segmentIncrement, midline, joints)
    
    print("Error: ", error, "Threshold: ", errorThreshold)

    if error < errorThreshold:
      increments += 1

      segmentEnd[0] += segmentIncrement
      print("")
      segmentEnd[1] = midline.iat[increments, 0] # !!! 0 should be the frame number 
    
    else:
      print("setting Sb to Se-Si and Se to Sb")
      segmentBeginning = segmentEnd.copy() 
      segmentBeginning[0] -= segmentIncrement
      segmentEnd = segmentBeginning.copy()
      joints.append(segmentBeginning)
      plt.scatter(increments, segmentEnd[1], color="green") #ls='solid'
      
  print("All segments built")

  return joints
  

def generateSegments(length, midline):
  joints = [[0 for x in range(3)] for y in range(0)]
  joints.append([0.0, 0.0, 0]) # contains x, y, and increment

  frames = len(midline[0])
  print("Len midline: ", frames)

  incrementLength = length/200 # a 200th of the fish length
  segmentBeginning = [0, 0]
  segmentEnd = [0, 0]
  increments = 0
  error = 0

  while increments < len(midline): # number of rows
    error = 0

    for f in range(len(midline[0])): # number of columns
      
      segmentBeginning[0] = midline[joints[len(joints)-1][2]][f][0]
      segmentBeginning[1] = midline[joints[len(joints)-1][2]][f][1]

      segmentEnd[0] = midline[increments][f][0]
      segmentEnd[1] = midline[increments][f][1]

      a = abs((segmentEnd[1]-segmentBeginning[1]) * midline[increments][f][0] - (segmentEnd[0]-segmentBeginning[0]) * midline[increments][f][1] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1])
      b = math.sqrt((segmentEnd[0]-segmentBeginning[0])**2 + (segmentEnd[1] - segmentBeginning[1])**2)
      
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
        joints.append([midline[increments][0][0], midline[increments][0][1], increments])
        print("Adding joint: ", joints[len(joints)-1])
      
      print("Error: ", error)
    
    else:
      print("Houston, we have a problem")

  return joints


def calculateSegmentError(segmentBeginning, segmentEnd, segmentIncrement, midline, totalIncrements, joints):
  error = 0
  print("\n")
  frames = midline.shape[1]

  #totalIncrements = int((segmentEnd[0] - segmentBeginning[0])/segmentIncrement)

  # segment[x, y], midline.iat[row, column]

  for i in range(2): # was frames, just going to do 1 for testing

    # while building, Se[0(x)] = sum of increments (later include the angle of midline calculated for), 
    #                 Se[1(y)] = My[total no. of increments]
    # so  Sb[1(y)] needs to start at My[0], 
    # 
    # After error, Sb[0(x)] needs to be Se[0(x)] and Sb[1(y)] then is My[total no. of increments]
    #     

    for j in range (totalIncrements):

      print("========= Debug: =========")
      print("segmentBeggining: ", segmentBeginning)
      print("segmentEnd: ", segmentEnd)
      print("midline cell at (", i, ", ", j, ") = ", midline.iat[j, i])
      
      # something is wrong with the math. the first value was -1.3755. a = -0.003303. b = 0.002401. Sb=[0.0006, -2.487] (but edited to be [0, -2.487]). Se=[0.0006, -2.487] 
      a = (segmentEnd[1]-segmentBeginning[1]) * midline.iat[j, i] - (segmentEnd[0]-segmentBeginning[0]) * midline.iat[j, i] + segmentBeginning[0] * segmentEnd[1] - segmentEnd[0] * segmentBeginning[1]
      
      b = math.sqrt((segmentEnd[0]-segmentBeginning[0])**2 + (segmentEnd[1] - segmentBeginning[1])**2)

      if (error < abs(a/b)):
        error = abs(a/b)/100

      if (error > errorThreshold):
        #print("Error level reached
        return error

      print("Frame ", i, ", increment ", j, " error: ", error)

  return error
  
  
# implementation of equal segments
def createEqualSegments(segmentCount, length):
  joints = []
  segmentLength = (length)/segmentCount
  segmentBeginning = 0
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
  
  
def main():
  fishMidline = loadMidlineData("Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls")
  
  length = 0.102

  #joints = createJointsGeneration(length, fishMidline)

  joints = generateSegments(length, fishMidline)
  
  print("==========================")

  print("size of joints = ", len(joints),", Joints: ", joints)

  for j in range(len(joints)):
    plt.plot(joints[j])#joints[j][0], joints[j][1])

  plt.show()

  plt.show()

main()