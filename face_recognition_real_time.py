import cv2
from face_detection import face_detector, draw_boxes
import pickle
from face_recognition import face_recognizer

model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = 'deploy.prototxt'

recognizer_pickle_in = open('recognizer.pickle', 'rb')
recognizer = pickle.load(recognizer_pickle_in)

label_encoder_pickle_in = open('le.pickle', 'rb')
label_encoder = pickle.load(label_encoder_pickle_in)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()

    detections, resized_image = face_detector(prototxt_path, model_path, image)
    result_image = face_recognizer(detections, resized_image)

    cv2.imshow('Video', result_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()