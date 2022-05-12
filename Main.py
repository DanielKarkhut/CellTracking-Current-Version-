# Daniel Karkhut

import cv2
import copy
import os
from detectors import Detectors
from tracker import Tracker
from cell import Cell
from input import Input

# Create Input Object
inp = Input()

# Create Object Detector
blur, dilate, cellSize = inp.getBlurAndDilate()
detector = Detectors(blur, dilate, 1)

# Open Dialog box for video folder and cell saving folder
cameraFolder = inp.getFolderLocation("Please pick a video folder")
resultsFolder = inp.getFolderLocation("Please pick a folder for your results")

# Create Object Tracker
tracker = Tracker(100, 2, 5000, 100)

traceStart = None #set 
traceEnd = None #set

currFrame = 0

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]
#images = []
# Loop Through Contents of TiffFolder
for f in sorted(os.listdir(cameraFolder)):
  if f.endswith(".tiff"):
    # Capture frame-by-frame
    frame = cv2.imread(cameraFolder + "/" + f)
    
    print(currFrame)

    if frame is None:
        currFrame = currFrame + 1
        break

    frame = cv2.flip(frame, 1)

    #images.append(frame)

    # Make copy of original frame
    orig_frame = copy.copy(frame)

    if (traceStart is None):
      traceStart = (int)(2*frame.shape[1]/6)
      traceEnd = (int)(4*frame.shape[1]/6)


    # Detect and return centeroids of the objects in the frame
    centers = detector.Detect(frame)

    # If centroids are detected then track them
    if (len(centers) > 0):

        # Track object using Kalman Filter
        tracker.Update(centers)

        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            if (len(tracker.tracks[i].trace) > 1):
                for j in range(len(tracker.tracks[i].trace)-1):
                    #Draw trace lines and beginning/end lines
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)
                    cv2.line(frame, (traceStart, 0), (traceStart, 1000), (255, 0, 0), 2)
                    cv2.line(frame, (traceEnd, 0), (traceEnd, 1000), (255, 0, 0), 2)

                    # Here means: tracker.tracks[which cell is trace being drawn for].trace[index of position of cell at specific frame][0 = x, 1 = y][unsure]
                    # if cell trace begins before x = 200 and ends after x = 1000, save the trace and store the cell
                    # used to save cell's movement
                    if ( (tracker.tracks[i].trace[0][0][0] < traceStart) and (tracker.tracks[i].trace[len(tracker.tracks[i].trace)-1][0][0] > traceEnd) and (tracker.tracks[i].tracked == 0) ):
                        #Make directory for new cell
                        cellDirectory = resultsFolder + "/Cell_{}".format(tracker.tracks[i].track_id) 
                        os.mkdir(cellDirectory)
                        
                        # create intermediary cell object
                        cell = Cell(tracker.tracks[i].track_id, cellDirectory)

                        # loop to capture full movement of cell
                        for x in range(len(tracker.tracks[i].trace)):
                            cell.updateValues(tracker.tracks[i].trace[x][0][0], tracker.tracks[i].trace[x][1][0])

                            if (cell.indexFirstLineisPassed is None) and (tracker.tracks[i].trace[x][0][0] >= traceStart):
                                cell.indexFirstLineisPassed = x 

                            if (cell.indexMiddleLineisPassed is None) and (tracker.tracks[i].trace[x][0][0] >= ((traceStart + traceEnd)/2)):
                                cell.indexMiddleLineisPassed = x

                        # take last photo 
                        k = currFrame 
                        photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        #photoFrame = images[k] USE INSTEAD OF LINE ABOVE IF FILES NOT INCREMENTING BY 1
                        crop = photoFrame[(int(tracker.tracks[i].trace[-1][1][0])  - int(cellSize/2)):(int(tracker.tracks[i].trace[-1][1][0]) + int(cellSize/2)), (int(tracker.tracks[i].trace[-1][0][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[-1][0][0]) + int(cellSize/2))].copy()
                        cell.lastCrop = crop

                        # take first photo
                        k = currFrame - (len(tracker.tracks[i].trace) + tracker.tracks[i].skipped_frames) + cell.indexFirstLineisPassed       
                        photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        #photoFrame = images[k] USE INSTEAD OF LINE ABOVE IF FILES NOT INCREMENTING BY 1
                        crop = photoFrame[(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][1][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][1][0]) + int(cellSize/2)), (int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][0][0]) - int(cellSize/2)):(int(tracker.tracks[i].trace[cell.indexFirstLineisPassed + tracker.tracks[i].skipped_frames][0][0]) + int(cellSize/2))].copy()
                        cell.firstCrop = crop

                        #take middle photo
                        k = currFrame - (len(tracker.tracks[i].trace) + tracker.tracks[i].skipped_frames) + cell.indexMiddleLineisPassed
                        photoFrame = cv2.imread(cameraFolder + "/" + str(k).zfill(6) + ".tiff") 
                        #photoFrame = images[k] USE INSTEAD OF LINE ABOVE IF FILES NOT INCREMENTING BY 1
                        crop = photoFrame[(0):(photoFrame.shape[0]), (int(tracker.tracks[i].trace[cell.indexMiddleLineisPassed][0][0]) - int(cellSize)):(int(tracker.tracks[i].trace[cell.indexMiddleLineisPassed][0][0]) + int(cellSize))].copy()
                        cell.midCrop = crop

                        # remove cell from being tracked again by setting initial position high
                        tracker.tracks[i].tracked = 1

                        cell.generateVelGraph()
                        cell.saveImage()

                        del cell
                # Display the resulting tracking frame
                cv2.imshow('Tracking', frame)

            # Display the original frame
            cv2.imshow('Original', orig_frame)

            # Slower the FPS
            cv2.waitKey(100)
    # Increment Frame Number
    currFrame = currFrame + 1