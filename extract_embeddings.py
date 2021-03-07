import cv2
import os
import imutils
import numpy as np
import pickle
from face_detection import face_detector


def extract_embeddings():
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    prototxt_path = 'deploy.prototxt'

    # extract features of face - open face embedder
    embedder = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')

    embeddings = []
    names = []
    faces_count = 0

    # facial embeddings of each face
    for dir_path, sub_dir_names, file_names in os.walk('faces'):
        for file_name in file_names:
            name = os.path.basename(dir_path)
            image_path = os.path.join(dir_path, file_name)
            print('Name: ', name)
            print('Image path:', image_path)

            image = cv2.imread(image_path)

            detections, resized_image = face_detector(prototxt_path, model_path, image)

            print("detections: ---------------------------------------")
            print(detections.shape[2])


            # each image must have at least one face detected
            # detections.shape[2]
            if len(detections) > 0:
                # one face only - with highest confidence level
                max_conf = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, max_conf, 2]

                if confidence > 0.5:
                    # bounding box of face
                    h, w = resized_image.shape[:2]  # exclude no. of channels
                    box = detections[0, 0, max_conf, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")  # numpy array values to int
                    print(box)

                    # region of interest - face
                    roi = resized_image[y1:y2, x1:x2]
                    roiH, roiW = roi.shape[:2]
                    print(roiH, roiW)

                    # embed if roi is large enough
                    if roiH >= 20 or roiW >= 20:
                        # Image BLOB
                        roiBlob = cv2.dnn.blobFromImage(roi, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

                        # pass to cnn - converts to 128-d vector (describes the face)
                        embedder.setInput(roiBlob)
                        vector = embedder.forward()

                        print(type(vector))

                        names.append(name)
                        embeddings.append(vector.flatten())
                        faces_count += 1

                        print(faces_count)
                        print(names)
                        print(embeddings)

    # pickling embeddings and names
    dict = {
        'embeddings': embeddings,
        'names': names
    }

    print(dict)
    pickle_out = open('face_embeddings.pickle', 'wb')
    pickle.dump(dict, pickle_out)
    pickle_out.close()

    # unpickling - read dict in bytes from pickle file
    pickle_in = open('face_embeddings.pickle', 'rb')
    extracted_dict = pickle.load(pickle_in)
    print(type(extracted_dict['embeddings'][0]))
    print(faces_count)


extract_embeddings()


# image = cv2.imread('faces/Jimin/jimin01.jpg')
# resized_image = imutils.resize(image, width=600)
# h, w = resized_image.shape[:2]
# cv2.imshow('resized image', resized_image)
#
# cv2.waitKey(0)
