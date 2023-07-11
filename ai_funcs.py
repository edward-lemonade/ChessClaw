import cv2
from keras.applications.vgg16 import VGG16, decode_predictions
from numpy import median, pi, reshape, array, linalg
import math

def iCap(cam = 1):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise Exception(f"failed to read frame from camera {cam}")
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        return frame
    finally:
        cv2.destroyAllWindows()
        cap.release()

def predict(frame):
    image = cv2.resize(frame, (224, 224))
    image = image.reshape(1, 224, 224, image.shape[2])
    model = VGG16()
    p = model.predict(image)
    label = decode_predictions(p)[0][0][1]

    return label

def CannyEdges(frame, sigma=0.33):
    # step 1 gray blur to reduce noise
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (5,5))

    # step 2 gradient calculation
    gradient = median(image)
    lower = int(max(0, (1.0 - sigma) * gradient))
    upper = int(min(255, (1.0 + sigma) * gradient))
    return image, cv2.Canny(image, lower, upper)

def HoughLine(edges, minHeight=100, maxGap=10):
    lines = cv2.HoughLines(edges, 1, pi / 180, 125, minHeight, maxGap)
    lines = reshape(lines, (-1, 2))
    return lines

def GridDetection(lines):
    h, v = [], []
    for r, t in lines:
        if t < pi / 4 or t > pi - pi / 4:
            v.append([r, t])
        else:
            h.append([r, t])

    return h, v

def Intersections(h, v):
    points = []
    for rh, th in h:
        for rv, tv in v:
            a = array([[math.cos(th), math.sin(th)], [math.cos(tv), math.sin(tv)]])
            b = array([rh, rv])
            x, y = linalg.solve(a,b)
            points.append((int(x),int(y)))
    return points
