from ai_funcs import iCap, predict, CannyEdges, HoughLine, GridDetection, Intersections
import cv2
import numpy
import math

pic = iCap()
image, edges = CannyEdges(pic)
# print(edges)
# print(numpy.shape(edges))
lines = HoughLine(edges)
# print(lines)
h, v = GridDetection(lines)
# print(h)
# print(v)
points = Intersections(h, v)
# print(points)


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