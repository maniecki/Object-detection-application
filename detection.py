import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

def find_contours(image, min_area=1000, max_area=float('inf')):
  
    contours_list = []
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # Aproksymacja konturów

            x, y, width, height = cv2.boundingRect(approx)              # Współrzędne prostokątu otaczającego konturę
            cx, cy = x + (width//2), y + (height//2)                    # Współrzędne środka
            contours_list.append({"cnt": contour, "area": area, "bbox": [x, y, width, height], "center": [cx, cy]})

    return contours_list

def contours_preprocessing(img, thresh1=20, thresh2=120, blur=5):

    image = cv2.GaussianBlur(img, (9,9),blur)
    image = cv2.Canny(image,thresh1,thresh2)                    # Binaryzacja obrazu

    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)             # Pogrubienie konturów
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)    # Zamknięcie konturów

    return image

def tflite_images_detection(img, interpreter, labels, sensor, sensor_width, min_conf=0.6):

    sensor_width += 20

    # Szczegóły modelu
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    detections_image = img.copy()
    image_rgb = cv2.cvtColor(detections_image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = detections_image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if float_input:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    detected_object = dict(object_name="", score=0, xmin=0, ymin=0, xmax=0, ymax=0)

    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(detections_image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i]*100)}%'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.putText(detections_image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(detections_image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            x_center = (xmax-xmin)//2 + xmin

            if x_center > (sensor-sensor_width/2) and x_center < (sensor+sensor_width/2):
                detected_object = dict(object_name=object_name, score=scores[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    return (detections_image, detected_object)

def load_model(model_path, labelmap_path):

     # Załadowanie labelmap
    with open(labelmap_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Załadowanie modelu TFLite do pamięci
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return (interpreter, labels)