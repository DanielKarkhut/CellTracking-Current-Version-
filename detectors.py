'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2

# set to 1 for pipeline images
debug = 1


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self, blurFactor, dilateFactor, analysis):
        """Initialize variables used by Detectors class
        Args:
            blurFactor: Degree of Gaussian Blur
            dilateFactor: Degree of contour dilation
            analysis: 0 to calculate deformation ratio, 1 to turn off
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.blurFactor = blurFactor #default 1
        self.dilateFactor = dilateFactor #default 3
        self.analysis = analysis

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)
        fgmask=cv2.GaussianBlur(fgmask, (self.blurFactor, self.blurFactor), 0)
        fgmask = cv2.dilate(fgmask, None, iterations=self.dilateFactor) # when we apply the blur, details get lost. So, the cell detail is getting lost, losing a well-defined contour of the cell so we are padding it (adding pixel value)
        fgmask=cv2.threshold(fgmask, 1, 255, cv2.THRESH_BINARY)[1]

        # Find contours
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        centers = []  # vector of object centroids in a frame
        # we only care about centroids with size of bug in this example
        # recommended to be tunned based on expected object size for
        # improved performance
        blob_radius_thresh = 9
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if (radius > blob_radius_thresh):
                    cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)

                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        return centers

        