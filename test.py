from ai_funcs import iCap, predict, CannyEdges, HoughLine, GridDetection, Intersections, ClusterPoints, augment_points
import cv2
import numpy
import math

TR = 5
RR = 1
ML = 100
MG = 10
SM = 0.33
BL = 5
SKIP = 3

def showLines(frame):
    global TR, RR, SM, BL, ML, MG, SKIP
    image, edges = CannyEdges(frame, sigma=SM, blur=BL)
    lines = HoughLine(edges, thetaRes=TR, rhoRes=RR, minLen=ML, maxGap=MG)
    h, v = GridDetection(lines, skip = SKIP)

    for r, t in v + h:
        a, b = math.cos(t), math.sin(t)
        x0, y0 = a * r, b * r
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000 * a))
        cv2.line(frame, pt1, pt2, (64,77,255), 3, cv2.LINE_AA)

    cv2.putText(frame, f"thetaRes: {TR} (w/s), rhoRes: {RR} (a/d)", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)
    cv2.putText(frame, f"minLength: {ML} ([/]), maxGap: {MG} (j/k)", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)
    cv2.putText(frame, f"sigma: {SM} (</>), blur: {BL} (+/-), skip: {SKIP} (n/m)", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)

    return frame

def updateHoughLineParams(k):
    global TR, RR, SM, BL, ML, MG, SKIP
    if k & 0xFF == ord('w'):
        TR = TR + 1
    elif k & 0xFF == ord('s'):
        TR = TR - 1
    elif k & 0xFF == ord('a'):
        RR = RR - 1
    elif k & 0xFF == ord('d'):
        RR = RR + 1
    elif k & 0xFF == ord('-'):
        BL = BL - 1
    elif k & 0xFF == ord('='):
        BL = BL + 1
    elif k & 0xFF == ord('<'):
        SM = SM - 0.01
    elif k & 0xFF == ord('>'):
        SM = SM + 0.01
    elif k & 0xFF == ord('['):
        ML = ML - 1
    elif k & 0xFF == ord(']'):
        ML = ML + 1
    elif k & 0xFF == ord('j'):
        MG = MG - 1
    elif k & 0xFF == ord('k'):
        MG = MG + 1
    elif k & 0xFF == ord('n'):
        SKIP = SKIP - 1
    elif k & 0xFF == ord('m'):
        SKIP = SKIP + 1

pic = iCap(render=showLines, interval=5, change=updateHoughLineParams)
image, edges = CannyEdges(pic, sigma=SM, blur=BL)
lines = HoughLine(edges, thetaRes=TR, rhoRes=RR, minLen=ML, maxGap=MG)
# print(lines)
h, v = GridDetection(lines, skip = SKIP)
# print(h)
# print(v)
points = Intersections(h, v)
# print(points)

points = ClusterPoints(points)
# points = augment_points(points)

for x, y in points:
    cv2.circle(image, (x,y), radius=5, color=(0,0,255), thickness=-1)
""" for r, t in v + h:
    a, b = math.cos(t), math.sin(t)
    x0, y0 = a * r, b * r
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000 * a))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000 * a))
    cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
 """
cv2.imshow("image", image)
cv2.imshow("edges", edges)
while cv2.waitKey(1) & 0xFF != ord(' '):
    pass

cv2.destroyAllWindows()