import cv2
from numpy import median, pi, reshape, array, linalg, mean, shape, empty, vstack
import math
from collections import defaultdict
import ckwrap
import traceback
import sys

"""
iCap is an interactive frame capture function used for capturing raw training
data, or actual data for inference. Press space for capturing, and 'q' for exit.

Parameters:
 - render: a function adds visual hue into the captured frame
 - cam: the camera id, defaults to 1
 - interval: capturing interval, defaults to 10ms
 - change: a function accepting key stroke to adjust parameters

Return:
 - ret: the data structure the render function generated based on the frame
 - frame: the captured frame

"""
def iCap(render = None, cam = 1, interval = 10, change = None):
    if type(cam) == int:
        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    else:
        cap = cv2.VideoCapture(cam)
    try:
        ready = False
        while True:
            ret = None
            success, frame = cap.read()
            if not success:
                raise Exception(f"failed to read frame from camera {cam}")
            
            if render:
                try:
                    ret, rendered = render(frame.copy())
                except Exception:
                    print(traceback.format_exc())
                    rendered = frame
            else:
                rendered = frame
            cv2.imshow("frame", rendered)

            k = cv2.waitKey(interval)
            if k & 0xFF == ord(' '):
                ready = True
            elif k & 0xFF == ord('q'):
                return None, None
            elif k and change:
                change(k)

            if ready and ret is not None:
                break

        return ret, frame
    finally:
        cv2.destroyAllWindows()
        cap.release()

"""
CannyEdges is a wrapper function to invoke cv2 Canny function to generate a
frame with only canny edges from the source image. It includes steps such as
turning the image into grayscale, then blur, calculate the gradient. Then
ignore details without obvious brightness change.

Parameters:
 - frame: source image
 - sigma: the scale how much the details are simplified
 - blur: how blury before using the canny edge algorithm

Returns:
 - image: the blurred grayscale image
 - canny: the image with only canny edges
"""
def CannyEdges(frame, sigma=0.33, blur=5):
    # step 1 gray blur to reduce noise
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (blur,blur))

    # step 2 gradient calculation
    gradient = median(image)
    lower = int(max(0, (1.0 - sigma) * gradient))
    upper = int(min(255, (1.0 + sigma) * gradient))
    return image, cv2.Canny(image, lower, upper)

"""
HoughLine function is a wrapper to cv2 HoughLines function. It detects
straightlines from the given canny edge image, and return a list of lines in
(rho, theta) notion.

Parameters:
 - edges: the canny edge image
 - thetaRes: the theta resolution
 - rhoRes: the rho resolution
 - minLen: minimun required length of the line
 - maxGap: maximum allowed gap in the line

Returns:
 - lines: two dimension array, each row is a pair of rho and theta.
"""
def HoughLine(edges, thetaRes = 1, rhoRes = 1, minLen=140, maxGap=10):
    lines = cv2.HoughLines(edges, rhoRes, thetaRes * pi / 180, minLen, maxGap)
    lines = reshape(lines, (-1, 2))
    return lines

"""
GridLines split the group of lines into horizontal lines and vertical lines
"""
def GridLines(lines):
    h, v = [], []
    for r, t in lines:
        if t < pi / 4 or t > pi - pi / 4:
            v.append([r, t])
        else:
            h.append([r, t])
 
    return h, v

"""
Intersections calculate all the intersections between two groups of lines

Parameters:
 - h: horizontal lines
 - v: vertical lines

Returns:
 - points: a list of points in (x, y) format
"""
def Intersections(h, v):
    points = []
    for rh, th in h:
        for rv, tv in v:
            a = array([[math.cos(th), math.sin(th)], [math.cos(tv), math.sin(tv)]])
            b = array([rh, rv])
            x, y = linalg.solve(a,b)
            points.append((int(x),int(y)))
    return points

"""
FitToGrid groups the intersection points into eleven rows and each with eleven
clusters. Take the mean of the coordinates of each cluster, we now have 11x11
grid line intersections. We use 1 dimension K-Means algorithm implemented in
ckwrap library to group and cluster the points. This assumes that the grid is
aligned horizontally.

Parameters:
 - points: a list of points in (x, y) format

Returns:
 - grid: a 2 dimension array of points (x, y), rows then columns

"""
def FitToGrid(points):
    rowLabels = ckwrap.ckmeans(points.T[1], 11).labels
    rows = [ empty([0, 2]) for i in range(11) ]
    for i in range(len(points)):
        rows[rowLabels[i]] = vstack([rows[rowLabels[i]], points[i]])

    clusterLabels = list(map(lambda r: ckwrap.ckmeans(r.T[0], 11).labels, rows))
    clusters = [ [ empty([0, 2]) for i in range(11) ] for j in range(11) ]
    for i in range(11):
        for j in range(len(rows[i])):
            clusters[i][clusterLabels[i][j]] = vstack([clusters[i][clusterLabels[i][j]], rows[i][j]])

    grid = empty([11, 11, 2])
    for i in range(11):
        for j in range(11):
            grid[i][j] = clusters[i][j].mean(axis=0)

    return grid.astype(int)
