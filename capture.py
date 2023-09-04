from ai_funcs import iCap, predict, CannyEdges, HoughLine, GridLines, Intersections, FitToGrid
import cv2
import numpy
import math

TR = 1
RR = 1
ML = 140
MG = 10
SM = 0.33
BL = 5

def showGrid(frame):
    global TR, RR, SM, BL, ML, MG
    image, edges = CannyEdges(frame, sigma=SM, blur=BL)
    lines = HoughLine(edges, thetaRes=TR, rhoRes=RR, minLen=ML, maxGap=MG)
    h, v = GridLines(lines)
    points = numpy.array(Intersections(h, v))
    grid = FitToGrid(points)

    for row in grid[1:-1]:
        v = numpy.diff(row[1:-1], axis=0).var(axis=0)
        assert v[0] < 40 and v[1] < 5
        for point in row[1:-1]:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(64, 77, 255), thickness=-1)

    cv2.putText(frame, f"thetaRes: {TR} (w/s), rhoRes: {RR} (a/d)", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)
    cv2.putText(frame, f"minLength: {ML} ([/]), maxGap: {MG} (j/k)", (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)
    cv2.putText(frame, f"sigma: {SM} (</>), blur: {BL} (+/-)", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 77, 255), 2, 2)

    return grid, frame

def updateHoughLineParams(k):
    global TR, RR, SM, BL, ML, MG
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

while True:
    g, pic = iCap(render=showGrid, interval=5, change=updateHoughLineParams)
    if pic is None:
        break

    rn = 1
    try:
        with open("counter.txt", "r") as cf:
            counter = int(cf.read())
    except:
        counter = 0

    try:
        for r in g[2:10]:
            for i in range(1,9):
                width = int(r[i+1][0] - r[i][0])
                height = int(1.5 * width)
                left = int(r[i][0])
                bottom = int(min(r[i][1], r[i+1][1]))
                cropped = pic[max(0, bottom - height):bottom, left: left + width]
                cv2.imwrite(f'./raw/image_{counter}_{rn}_{i}.jpeg', cropped)
                counter += 1
            rn = rn + 1
    finally:
        with open("counter.txt", "w") as cf:
            cf.write(f"{counter}")