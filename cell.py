# Daniel Karkhut
# represents cell object

import math
import scipy.signal
import statistics
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from detectors import Detectors
from tracker import Tracker
from input import Input

class Cell:

    def __init__(self, id, directory):
        self.id = id
        self.xCoordCenter = []
        self.yCoordCenter = []
        self.indexFirstLineisPassed = None #store index that the first tracking line is passed
        self.indexMiddleLineisPassed = None #store index for mid point passing
        self.directory = directory
        self.firstCrop = None
        self.midCrop = None
        self.lastCrop = None


    #Add coords to cell xCoordCenter and yCoordCenter
    def updateValues(self, xCoordCenter, yCoordCenter):
        self.xCoordCenter.append(int(xCoordCenter))
        self.yCoordCenter.append(int(yCoordCenter))

    #Create and save graph for cell
    def generateVelGraph(self):

        #stores amount of movement per frame
        diff = []

        for x in range(len(self.xCoordCenter)-1):
            #find the distance that was traveled using pythagorean theorem
            z = abs( math.sqrt( ((self.xCoordCenter[x+1] - self.xCoordCenter[x])** 2) + ((self.yCoordCenter[x+1] - self.yCoordCenter[x])** 2) ))
            #if data point is too high, replace with average value of array
            if (z > 100):
                diff.append(statistics.median(diff))
                continue
            
            #add to array
            diff.append(z)
        

        #apply smoothing to graph
        #window_length must be equal or less than len(diff) and must be odd
        window_length = len(diff)
        if window_length % 2 == 0: 
            xhat = scipy.signal.savgol_filter(diff, 7, 3) # even, window size 51, polynomial order 3
        else: 
            xhat = scipy.signal.savgol_filter(diff, 7, 3) # odd, window size, polynomial order 3

        #plotting
        plt.title("Speed vs Time") 
        plt.xlabel("Time (frames)") 
        plt.ylabel("Speed (mm)") 

        plt.plot(range(len(self.xCoordCenter)-1), xhat) 
    
        path = self.directory + "/VelGraph.png"
        plt.savefig(path)

        plt.clf()

        # generate time-velocity csv file
        a_list = np.array(range(len(self.xCoordCenter)-1))
        #a_list is time. xhat is velocity
        result = []
        temp = []
        count = 0
        for i in (range(len(a_list))):
            for j in range(2):
                if j == 1:
                    temp.append(self.xCoordCenter[i])
                if j == 0:
                    temp.append(a_list[i])
            result.append(temp)
            temp = []
            count += 1
        CSVpath = self.directory + "/t-v.csv"
        
        np.savetxt(CSVpath, result, delimiter=",", fmt = '%1.f')


        ####### DEFORMATION RATIO
        dst = cv2.cvtColor(self.midCrop, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(dst, 18, 50, None, 3)
        
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 60, None, 0, 0)

        top = 0 #top of image
        bottom = int(self.midCrop.shape[0]) #bottom of image
        center = int(self.midCrop.shape[0] / 2) #center of image

        topOfCell = 0
        bottomOfCell = bottom

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho

                #self.midCrop.shape[0] #height
                #self.midCrop.shape[1] #width
                #self.midCrop.shape[2] #color

                if (y0 < center) and (y0 > topOfCell): #find top position of cell (top edge of innerchannel)
                    topOfCell = y0

                if (y0 > center) and (y0 < bottomOfCell): #find bottom position of cell (top edge of innerchannel)
                    bottomOfCell = y0

        rightOfCell = 0
        leftOfCell = 10000
        for x in range(center-2, center+2): #finds max width of cell when squeezed between center 20 pixels 

            #find leftmost point
            for y in range(dst.shape[1]):
                if (dst[x][y] == 255) and (y < leftOfCell): #if pixel is white
                    leftOfCell = y
                    break


            #find rightmost point
            for y in reversed(range(dst.shape[1])):
                if (dst[x][y] == 255) and (y > rightOfCell): #if pixel is white
                    rightOfCell = y
                    break
        
        deformationRatio = (abs(topOfCell - bottomOfCell) / abs(rightOfCell - leftOfCell))

        txtPath = self.directory + "/DeformationRatio.txt"
        file = open(txtPath, "w+")
        file.write(str(deformationRatio))
        file.close()

    def saveImage(self):
        #if the height and length of the image =/= Image Size(cellSize) //// x.shape[0] is height, x.shape[1] is width
        try:
            #save First Crop
            if (  self.firstCrop.shape[0] !=  self.firstCrop.shape[1] ):
                blackImage = np.zeros((self.firstCrop.shape[1] -  self.firstCrop.shape[0],  self.firstCrop.shape[1], 3), float)
                self.firstCrop = np.concatenate((self.firstCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Firstcrop.png"
            cv2.imwrite(path, self.firstCrop)

            #save First Crop
            if (  self.midCrop.shape[0] !=  self.midCrop.shape[1] ):
                blackImage = np.zeros((self.midCrop.shape[1] -  self.midCrop.shape[0],  self.midCrop.shape[1], 3), float)
                self.midCrop = np.concatenate((self.midCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Midcrop.png"
            cv2.imwrite(path, self.midCrop)

            #save First Crop
            if (  self.lastCrop.shape[0] !=  self.lastCrop.shape[1] ):
                blackImage = np.zeros((self.lastCrop.shape[1] -  self.lastCrop.shape[0], self.lastCrop.shape[1], 3), float)
                self.lastCrop = np.concatenate((self.lastCrop, blackImage), axis=0) #first variable over second variable
            path = self.directory + "/Lastcrop.png"
            cv2.imwrite(path, self.lastCrop)
        except cv2.error as e:
            print("Cell not saved")
