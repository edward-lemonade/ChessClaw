from cv2 import VideoCapture, imshow, waitKey, resize, destroyAllWindows
from keras.applications.vgg16 import VGG16, decode_predictions

def iCap(cam = 1):
    cap = VideoCapture(cam)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise Exception(f"failed to read frame from camera {cam}")
            imshow("frame", frame)
            if waitKey(1) & 0xFF == ord(' '):
                break

        return frame
    finally:
        destroyAllWindows()
        cap.release()

def predict(frame):
    image = resize(frame, (224, 224))
    image = image.reshape(1, 224, 224, image.shape[2])
    model = VGG16()
    p = model.predict(image)
    label = decode_predictions(p)[0][0][1]

    return label
