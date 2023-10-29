from ai_funcs import iCap, CannyEdges, HoughLine, GridLines, Intersections, FitToGrid
import cv2
import numpy
import yolov5
from chessv1_label_map import chessv1_label_map as label_map
import bisect

TR = 1
RR = 1
ML = 140
MG = 10
SM = 0.33
BL = 5

HVX=70
HVY=10
VVX=10
VVY=70

def showGrid(frame):
    global TR, RR, SM, BL, ML, MG
    image, edges = CannyEdges(frame, sigma=SM, blur=BL)
    lines = HoughLine(edges, thetaRes=TR, rhoRes=RR, minLen=ML, maxGap=MG)
    h, v = GridLines(lines)
    points = numpy.array(Intersections(h, v))
    grid = FitToGrid(points)

    # for row in grid:
    #     for p in row:
    #         cv2.circle(edges, p, radius=5, color=(64,77,255), thickness=3)
    # cv2.imshow("edges", edges)

    for row in grid[1:-1]:
        v = numpy.diff(row[1:-1], axis=0).var(axis=0)
        assert v[0] < HVX and v[1] < HVY
        cv2.line(frame, row[1], row[-2], color=(64,77,255), thickness=3)

    for col in range(1,10):
        trend = numpy.diff(grid[1:-1,col], axis=0)
        v = numpy.diff(trend, axis=0).var(axis=0)
        assert v[0] < VVX and v[1] < VVY
        cv2.line(frame, grid[1,col], grid[9,col], color=(64,77,255), thickness=3)

    return grid, frame

model = yolov5.load("best.pt")

# set model parameters
model.conf = 0.55  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 40  # maximum number of detections per image

while True:
    g, pic = iCap(render=showGrid, interval=5, cam=1)
    if pic is None:
        break

    result = model(pic.copy())
    confi = result.pred[0][:,4]
    boxes = numpy.array(result.pred[0][:,:4]).astype(int)
    labels = list(map(lambda x:label_map[x], numpy.array(result.pred[0][:,5]).astype(int))
                  )
    bc = list(map(lambda b:(int((b[0] + b[2])/2), b[3] - int((b[3]-b[1])/4)) ,boxes))
    grid_loc = list(map(lambda c:(c[0], bisect.bisect(g.mean(axis=1)[:,1], c[1])),bc))
    grid_loc = list(map(lambda c:(bisect.bisect(g[c[1],:,0],c[0]), c[1]) , grid_loc))
    print(grid_loc)

    board = [ [ (" ",0) for j in range(8) ] for i in range(8)]
    for i in range(len(labels)):
        row = grid_loc[i][1] - 2
        col = grid_loc[i][0] - 2
        old, prob = board[row][col]
        if prob < confi[i]:
            if prob > 0:
                print(f"overriding at ({row}, {col}) from {old} to {labels[i]}, confi from {prob} to {confi[i]}")
            board[row][col] = (labels[i], confi[i])
        else:
            print(f"ignoring at ({row}, {col}) keeping {old} instead of {labels[i]}, confi {prob} vs {confi[i]}")
    print(board)

    lines = []
    for row in numpy.array(board)[:,:,0]:
        blank = 0
        line = ""
        for l in row:
            if l == " ":
                blank += 1
            else:
                if blank > 0:
                    line += f"{blank}"
                blank = 0
                line += l
        if blank > 0:
            line += f"{blank}"
        lines.append(line)
    FEN = "/".join(lines)
    print(FEN)

    for b in range(len(boxes)):
        cv2.circle(pic, bc[b], radius=5, color=(64,77,255), thickness=4)
        cv2.rectangle(pic, boxes[b,:2], boxes[b,2:], color=(235, 52, 82), thickness=3)
        cv2.putText(pic, labels[b], (boxes[b,0], boxes[b,1] - 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(64,77,255), thickness=2)
    cv2.imshow("detected", pic)
    while cv2.waitKey(100) & 0xFF != ord(' '):
        pass
    cv2.destroyAllWindows()