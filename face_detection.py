import numpy as np
import cv2
import imutils


def face_detector(prototxt_path, weight_path, image):
    # Load Caffe model
    print('loading model')
    detector = cv2.dnn.readNetFromCaffe(prototxt_path, weight_path)

    resized_image = imutils.resize(image, width=600)

    # Image BLOB (Binary Large Object) - prepare image for classification by the model
    # image, scaling, spatial size, mean subtraction
    blob = cv2.dnn.blobFromImage(cv2.resize(resized_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Face Detection
    print('computing face detections')
    detector.setInput(blob)
    detections = detector.forward()

    return detections, resized_image


def draw_boxes(image, detections):
    # faces detected
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(f'{i} confidence: {confidence}')
        # confidence threshold
        if confidence > 0.5:
            # bounding box
            h, w = image.shape[:2]  # exclude no. of channels
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print('-----------------')
            print(box)
            x1, y1, x2, y2 = box.astype("int")  # numpy array values to int
            # draw box around the face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


# def main():
#     model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
#     prototxt_path = 'deploy.prototxt'
#     image_path = 'bts_group4.jpg'
#
#     image = cv2.imread(image_path)
#     detections, image = face_detector(prototxt_path, model_path, image)
#
#     print(detections)
#     print(detections.shape)  # 4D - 0, 0, nth face, attributes
#
#     draw_boxes(image, detections)
#
#     h, w = image.shape[:2]
#     print(h, w)
#     print(type(h))
#     cv2.imshow("Result", image)
#     # cv2.imshow("Result", cv2.resize(image, (int(w/3), int(h/3))))  # accepts int
#     cv2.waitKey(0)
#
#
# main()
