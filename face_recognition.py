import cv2
import numpy as np
import pickle
import imutils

from sklearn import preprocessing, svm
from face_detection import face_detector


def generate_recognizer_and_labels():
    # load face embeddings
    face_embeddings_pickle_out = open('face_embeddings.pickle', 'rb')
    face_embeddings = pickle.load(face_embeddings_pickle_out)

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(face_embeddings['names'])
    print(labels)

    # train recognizer
    recognizer = svm.SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(face_embeddings["embeddings"], labels)

    # to pickle
    le_pickle_out = open('le.pickle', 'wb')
    pickle.dump(le, le_pickle_out)
    le_pickle_out.close()

    recognizer_pickle_in = open('recognizer.pickle', 'wb')
    pickle.dump(recognizer, recognizer_pickle_in)
    recognizer_pickle_in.close()


def face_recognizer(detections, image):

    # extract features of face - open face embedder
    embedder = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')

    recognizer_pickle_in = open('recognizer.pickle', 'rb')
    recognizer = pickle.load(recognizer_pickle_in)

    label_encoder_pickle_in = open('le.pickle', 'rb')
    label_encoder = pickle.load(label_encoder_pickle_in)

    print('--------label encoder---------')
    print(label_encoder.classes_)

    # faces detected
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        print(f'{i} confidence: {confidence}')

        # confidence threshold
        if confidence > 0.5:
            # bounding box
            h, w = image.shape[:2]  # exclude no. of channels
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print(box)
            x1, y1, x2, y2 = box.astype("int")  # numpy array values to int

            # region of interest - face
            roi = image[y1:y2, x1:x2]
            roiH, roiW = roi.shape[:2]

            # embed of roi
            if roiH >= 20 or roiW >= 20:
                try:
                    roiBlob = cv2.dnn.blobFromImage(roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                except:
                    break
                embedder.setInput(roiBlob)
                vector = embedder.forward()

                # classification (face recognition)
                predictions = recognizer.predict_proba(vector)[0]
                print('predictions--------------------------------')
                print(predictions)
                max_pred = np.argmax(predictions)
                probability = predictions[max_pred]
                print('max_pred--------------------------------')
                print(max_pred)
                name = label_encoder.classes_[max_pred]

                # draw box around the face
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # put name of face
                cv2.putText(image, str(name), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                cv2.putText(image, str(probability * 100), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    return image


generate_recognizer_and_labels();

# Test Face Recognizer
image_path = 'face_test_images/jimin.jpg'
image = cv2.imread(image_path)
cv2.imshow("Original", image)

model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = 'deploy.prototxt'

detections, resized_image = face_detector(prototxt_path, model_path, image)
result_image = face_recognizer(detections, resized_image)

cv2.imshow("Recognized Face", result_image)
cv2.waitKey(0)
