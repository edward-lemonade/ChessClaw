import cv2
from keras.applications.vgg16 import VGG16, decode_predictions
from numpy import median, pi, reshape, array, linalg, mean, shape, empty, vstack
import math
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
import ckwrap

def iCap(render = None, cam = 1, interval = 10, change = None):
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise Exception(f"failed to read frame from camera {cam}")

            if render:
                try:
                    rendered = render(frame.copy())
                except:
                    rendered = frame
            else:
                rendered = frame
            cv2.imshow("frame", rendered)

            k = cv2.waitKey(interval)
            if k & 0xFF == ord(' '):
                break
            if k and change:
                change(k)

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

def CannyEdges(frame, sigma=0.33, blur=5):
    # step 1 gray blur to reduce noise
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (blur,blur))

    # step 2 gradient calculation
    gradient = median(image)
    lower = int(max(0, (1.0 - sigma) * gradient))
    upper = int(min(255, (1.0 + sigma) * gradient))
    return image, cv2.Canny(image, lower, upper)

def HoughLine(edges, thetaRes = 1, rhoRes = 1, minLen=140, maxGap=10):
    lines = cv2.HoughLines(edges, rhoRes, thetaRes * pi / 180, minLen, maxGap)
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

def ClusterPoints(p):
    dists = spatial.distance.pdist(p)
    s = cluster.hierarchy.single(dists)
    f = cluster.hierarchy.fcluster(s, 15, 'distance')
    cld = defaultdict(list)
    for i in range(len(f)):
        cld[f[i]].append(p[i])
    clv = cld.values()
    cls = map(lambda arr: (int(mean(array(arr)[:,0])), int(mean(array(arr)[:,1]))), clv)
    return sorted(list(cls), key = lambda k: [k[1], k[0]])

def augment_points(points):
    points_shape = list(shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (int(rw_x[i]), int(y_mean))
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points

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

    return grid