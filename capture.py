from ai_funcs import iCap, CannyEdges, HoughLine, GridLines, Intersections, FitToGrid
import cv2
import numpy
import yolov5
from chessv1_label_map import chessv1_label_map as label_map
import bisect
from dotenv import load_dotenv
from os import getenv
from stockfish import Stockfish

load_dotenv()
PLAY_AS = getenv("STOCKFISH_PLAY_AS", "b")

TR = 1
RR = 1
ML = 140
MG = 10
SM = 0.1
BL = 3

HVX=70
HVY=10
VVX=10
VVY=70

best_move = ""

"""
updateSettings is passed to iCap to interactively change settings based on key stroke
"""
def updateSettings(k):
    global PLAY_AS
    if k & 0xFF == ord('f'):
        PLAY_AS = 'b' if PLAY_AS == 'w' else 'w'

"""
showGrid is passed to iCap as a render function to show the grid lines
in the video to guide the capture.
"""
def showGrid(frame):
    global TR, RR, SM, BL, ML, MG
    global best_move

    # below are the steps to detect the grid
    image, edges = CannyEdges(frame, sigma=SM, blur=BL)
    lines = HoughLine(edges, thetaRes=TR, rhoRes=RR, minLen=ML, maxGap=MG)
    h, v = GridLines(lines)
    points = numpy.array(Intersections(h, v))
    grid = FitToGrid(points)

    for row in grid:
        for p in row:
            cv2.circle(edges, p, radius=5, color=(64,77,255), thickness=3)
    cv2.imshow("edges", cv2.resize(edges, (800, 600)))

    # draw the horizontal lines
    for row in grid[1:-1]:
        v = numpy.diff(row[1:-1], axis=0).var(axis=0)
        # throw error if the variance of the distance between each columns is too big 
        assert v[0] < HVX and v[1] < HVY
        cv2.line(frame, row[1], row[-2], color=(64,77,255), thickness=3)

    # draw the vertical lines
    for col in range(1,10):
        trend = numpy.diff(grid[1:-1,col], axis=0)
        v = numpy.diff(trend, axis=0).var(axis=0)
        # throw error if the variance of the distance change between rows is too big
        assert v[0] < VVX and v[1] < VVY
        cv2.line(frame, grid[1,col], grid[9,col], color=(64,77,255), thickness=3)
        
    cv2.putText(frame, f"playing as: {PLAY_AS}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (64, 77, 255), 5, 2)
    cv2.putText(frame, f"best move: {best_move}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (64, 77, 255), 5, 2)
    return grid, cv2.resize(frame, (800, 600))

# load the fine-tuned yolov5 weights.
model = yolov5.load("best.pt")

# set model parameters
model.conf = 0.55  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 40  # maximum number of detections per image

ce_options = {
    "Threads": int(getenv("STOCKFISH_THREADS", "2")),
    "Debug Log File": "stockfish.log",
}

ce = None
# initialize, Stockfish(path=getenv("STOCKFISH_BINARY"), depth=int(getenv("STOCKFISH_DEPTH", "10")), parameters=ce_options)

while True:
    g, pic = iCap(render=showGrid, interval=5, cam=1, change=updateSettings)
    # when space or 'q' is pressed, iCap returns
    if pic is None:
        # 'q' is pressed, iCap return None for the pic, quit
        break

    # invoke inferernce
    result = model(pic.copy())
    # confi is the confidence level of each detected obj
    confi = result.pred[0][:,4]
    # boxes is the bounding box of each detected obj
    boxes = numpy.array(result.pred[0][:,:4]).astype(int)
    # convert the raw label into the FEN character based on a label_map
    labels = list(map(lambda x:label_map[x], numpy.array(result.pred[0][:,5]).astype(int)))
    # calculate the estmated bottom center position of the piece
    bc = list(map(lambda b:(int((b[0] + b[2])/2), b[3] - int((b[3]-b[1])/4)) ,boxes))
    # convert the y coordinate into the board location 0-12 using bisect
    grid_loc = list(map(lambda c:(c[0], bisect.bisect(g.mean(axis=1)[:,1], c[1])),bc))
    # convert the x coordinate into the board location 0-12 using bisect
    grid_loc = list(map(lambda c:(bisect.bisect(g[c[1],:,0],c[0]), c[1]) , grid_loc))
    print(grid_loc)

    # create an empty board
    board = [ [ (" ",0) for j in range(8) ] for i in range(8)]

    # put the pieces into the board
    for i in range(len(labels)):
        row = grid_loc[i][1] - 2
        col = grid_loc[i][0] - 2
        old, prob = board[row][col]
        # if multiple pieces detected in the same location, use the one with higher confidence level
        if prob < confi[i]:
            if prob > 0:
                print(f"overriding at ({row}, {col}) from {old} to {labels[i]}, confi from {prob} to {confi[i]}")
            board[row][col] = (labels[i], confi[i])
        else:
            print(f"ignoring at ({row}, {col}) keeping {old} instead of {labels[i]}, confi {prob} vs {confi[i]}")
    print(board)

    # convert the board into FEN format line by line
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
    FEN = FEN + " " + PLAY_AS + " - - 1 2"
    print(FEN)

    # draw the boxes and piece labels in the picture and show it for validation, press space to continue
    for b in range(len(boxes)):
        cv2.circle(pic, bc[b], radius=5, color=(64,77,255), thickness=4)
        cv2.rectangle(pic, boxes[b,:2], boxes[b,2:], color=(235, 52, 82), thickness=3)
        cv2.putText(pic, labels[b], (boxes[b,0], boxes[b,1] - 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(64,77,255), thickness=2)
    cv2.imshow("detected", pic)
    while True:
        ki = cv2.waitKey(100) & 0xFF
        if ki == ord(' '):
            try:
                if ce is None:
                    ce = Stockfish(path=getenv("STOCKFISH_BINARY"), depth=int(getenv("STOCKFISH_DEPTH", "10")), parameters=ce_options)
                ce.set_fen_position(FEN)
                best_move = ce.get_best_move_time(2000)
            except:
                ce = None
                best_move = "invalid frame"
            break
        elif ki == ord('r'):
            best_move = "retrying..."
            break
    cv2.destroyAllWindows()
