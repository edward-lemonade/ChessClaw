import cv2
import os
import glob

try:
    with open("classified.txt", "r") as pf:
        ptr = int(pf.read())
except:
    ptr = 0

escape = False
while not escape:
    nextImage = glob.glob(f"./raw/image_{ptr}_*.jpeg")[0]
    ptr += 1
    image = cv2.imread(nextImage)

    name = ""
    submitted = False
    skipped = False
    while not submitted and not escape and not skipped:
        size = image.shape
        displayed = cv2.resize(image, (size[1] * 3, size[0] * 3))
        cv2.putText(displayed, f'>{name}_', (3,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64,77,255), 2, 2)
        cv2.imshow("piece", displayed)
        k = cv2.waitKey(30)
        c = k & 0xFF
        if c >= ord('a') and c<=ord('z') or c==ord(' '):
            name += chr(c)
        elif c == 8:
            name = name[:-1]
        elif c == 13:
            submitted = True
        elif c==27:
            escape = True
        elif c==9:
            skipped = True

    if not escape:
        if not skipped:
            with open("processed.csv", "a") as prf:
                prf.write(f'"{nextImage}","{name}"\n')
        with open("classified.txt", "w") as pf:
            pf.write(f"{ptr}")