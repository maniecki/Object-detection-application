from PyQt5 import uic
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from detection import find_contours, contours_preprocessing, tflite_images_detection, load_model

# Wątek do obsługi kamery i detekcji
class CameraThread(QThread):
    mainChangemap = pyqtSignal('QImage')
    contChangemap = pyqtSignal('QImage')
    objChangemap = pyqtSignal('QImage')
    detValues = pyqtSignal(dict)
    lastObj = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        global labels
        self.thresh1 = 0
        self.thresh2 = 0
        self.blur = 0
        
        self.is_counting = False
        
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(camIndex)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        resize_factor = 0.75
        sensor = 1000
        resized_sensor = sensor*resize_factor
        sensor_width = 60

        current_detections = []
        objImage = None
        
        while self.ThreadActive:
            try:
                ret, frame = cap.read()
                if ret:
                    resized_frame = cv2.resize(frame, (int(frame.shape[1]*resize_factor),int(frame.shape[0]*resize_factor)))
                    framePre = contours_preprocessing(resized_frame, self.thresh1, self.thresh2, self.blur)
                    contours = find_contours(framePre, min_area=400)

                    cv2.line(framePre, (int(resized_sensor-sensor_width/2),0), (int(resized_sensor-sensor_width/2), int(frame.shape[1]*resize_factor)), (255,255,255), 3)
                    cv2.line(framePre, (int(resized_sensor+sensor_width/2),0), (int(resized_sensor+sensor_width/2), int(frame.shape[1]*resize_factor)), (255,255,255), 3)

                    detections_image, detected_obj = tflite_images_detection(frame, interpreter, labels, sensor, sensor_width)

                    for contour in contours:
                        if contour["center"][0] > (resized_sensor-sensor_width/2) and contour["center"][0] <= (resized_sensor+sensor_width/2):
                            if detected_obj["object_name"] != '':
                                current_detections.append(detected_obj["object_name"])
                                self.lastObj.emit(detected_obj["object_name"])

                                o = frame[detected_obj["ymin"]:detected_obj["ymax"], detected_obj["xmin"]:detected_obj["xmax"]].copy()
                                objImage = cv2.resize(o, (120, 120))
                                objConvert = QImage(objImage.data, objImage.shape[1], objImage.shape[0], QImage.Format_BGR888)
                                objPic = objConvert.scaled(120, 120, Qt.KeepAspectRatio)
                                self.objChangemap.emit(objPic)
                                
                        elif contour["center"][0] >= (resized_sensor+sensor_width/2)+5 and contour["center"][0] < (resized_sensor+sensor_width/2)+35:
                            if current_detections:
                                count_det = {i:current_detections.count(i) for i in current_detections}
                                obj = max(count_det, key=count_det.get)

                                if self.is_counting:
                                    self.values[obj] += 1
                                    self.detValues.emit(self.values)
                                    
                                count_det = {}

                            current_detections.clear()
                    
                    cv2.line(detections_image, (int(sensor-sensor_width/2),0), (int(sensor-sensor_width/2), int(frame.shape[1])), (0,255,0), 4)
                    cv2.line(detections_image, (int(sensor+sensor_width/2),0), (int(sensor+sensor_width/2), int(frame.shape[1])), (0,255,0), 4)
                    cv2.line(detections_image, (int(sensor+sensor_width/2)+30,0), (int(sensor+sensor_width/2)+25, int(frame.shape[1])), (0,0,255), 3)

                    mainConvert = QImage(detections_image.data, detections_image.shape[1], detections_image.shape[0], QImage.Format_BGR888)
                    mainPic = mainConvert.scaled(460, 260, Qt.KeepAspectRatio)
                    self.mainChangemap.emit(mainPic)
                    
                    contConvert = QImage(framePre.data, framePre.shape[1], framePre.shape[0], QImage.Format_Grayscale8)
                    contPic = contConvert.scaled(300, 170, Qt.KeepAspectRatio)
                    self.contChangemap.emit(contPic)

            except Exception as error:
                print(error)

        cap.release()
    
    def stop(self):
        self.ThreadActive = False
        self.quit()

    @pyqtSlot(tuple)
    def setPreCalibration(self, calValues):
        thresh1, thresh2, blur = calValues
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        self.blur = blur

    @pyqtSlot(bool)
    def isCounting(self, is_counting):
        self.is_counting = is_counting
    
    @pyqtSlot(dict)
    def setValues(self, values):
        self.values = values

class MainWindow(QMainWindow):
    SLValues = pyqtSignal(tuple)
    isCounting = pyqtSignal(bool)
    counterSet = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("GUI/GUI.ui", self)

        self.labelmap_path = "labelmap/labelmap.txt"
        # self.model_path = "TFModel/Tensorflow2/workspace/models/my_tflite_model/detect.tflite"
        self.model_path = "TF2Model/detect.tflite"
        self.interpreter, self.labels = load_model(model_path=self.model_path, labelmap_path=self.labelmap_path)
        
        global labels, interpreter
        labels = self.labels
        interpreter = self.interpreter

        self.cameras = QCameraInfo.availableCameras()
        self.CameraTypeCB.addItems([cam.description() for cam in self.cameras])
        self.StartFeedBTN.clicked.connect(self.startCamera)
        self.StopFeedBTN.clicked.connect(self.stopCamera)
        self.StopFeedBTN.setEnabled(False)

        self.StartCntBTN.clicked.connect(self.startCounting)
        self.StopCntBTN.clicked.connect(self.stopCounting)
        self.StopCntBTN.setEnabled(False)
        self.StartCntBTN.setEnabled(False)

        self.Thresh1SL.valueChanged.connect(self.updateSLValues)
        self.Thresh2SL.valueChanged.connect(self.updateSLValues)
        self.BlurSL.valueChanged.connect(self.updateSLValues)
        
        self.detValues = {x:0 for x in labels}
        self.updateObjectsTE()

        self.ResetCntBTN.clicked.connect(self.resetPopup)

    def updateSLValues(self):
        self.thresh1 = self.Thresh1SL.value()
        self.thresh2 = self.Thresh2SL.value()
        self.blur = self.BlurSL.value()
        self.SLValues.emit((self.thresh1, self.thresh2, self.blur))

    def updateObjectsTE(self):
        self.ObjectsTE.clear()
        for i in range(len(self.labels)):
            value = self.detValues[labels[i]]
            self.ObjectsTE.append(f'{self.labels[i]}: {value}')

    @pyqtSlot(dict)
    def setDetectionValues(self, detValues):
        self.detValues = detValues
        self.updateObjectsTE()

    @pyqtSlot(str)
    def setObjLBL(self, name):
        self.LastObjectLBL.setText(name)

    # Obraz główny
    @pyqtSlot('QImage')
    def setMainImage(self, image):
        try:
            self.MainFeedLBL.setPixmap(QPixmap.fromImage(image))
        except Exception as error:
            self.LogTE.append(f'Uwaga! Error: {error}')
    
    # Obraz kontur
    @pyqtSlot('QImage')
    def setContImage(self, image):
        try:
            self.ContFeedLBL.setPixmap(QPixmap.fromImage(image))
        except Exception as error:
            self.LogTE.append(f'Uwaga! Error: {error}')

    # Obraz obiektu
    @pyqtSlot('QImage')
    def setObjImage(self, image):
        try:
            self.ObjectImageLBL.setPixmap(QPixmap.fromImage(image))
        except Exception as error:
            self.LogTE.append(f'Uwaga! Error: {error}')

    # Metoda rozpoczęcia obrazu z kamery
    def startCamera(self):
        try:
            self.LogTE.append(f"{QDateTime.currentDateTime().toString('d-MM-yyyy  hh:mm:ss')}: Start kamery ({self.CameraTypeCB.currentText()})")
            self.StopFeedBTN.setEnabled(True)
            self.StartFeedBTN.setEnabled(False)
            self.StartCntBTN.setEnabled(True)

            global camIndex
            camIndex = self.CameraTypeCB.currentIndex()

            self.CamThread = CameraThread()
            self.CamThread.mainChangemap.connect(self.setMainImage)
            self.CamThread.contChangemap.connect(self.setContImage)
            self.CamThread.objChangemap.connect(self.setObjImage)
            self.CamThread.detValues.connect(self.setDetectionValues)
            self.CamThread.lastObj.connect(self.setObjLBL)
            self.isCounting.connect(self.CamThread.isCounting)
            self.SLValues.connect(self.CamThread.setPreCalibration)
            self.counterSet.connect(self.CamThread.setValues)

            self.updateSLValues()
            self.counterSet.emit(self.detValues)
            
            self.CamThread.start()

        except Exception as error:
            self.LogTE.append(f'Uwaga! Error: {error}')
    
    # Metoda zatrzymania obrazu z kamery
    def stopCamera(self):
        try:
            self.LogTE.append(f"{QDateTime.currentDateTime().toString('d-MM-yyyy  hh:mm:ss')}: Zatrzymanie kamery")
            self.StartFeedBTN.setEnabled(True)
            self.StopFeedBTN.setEnabled(False)
            self.CamThread.stop()

            self.StopCntBTN.setEnabled(False)
            self.StartCntBTN.setEnabled(False)

        except Exception as error:
            self.LogTE.append(f'Uwaga! Error: {error}')

    def startCounting(self):
        self.StartCntBTN.setEnabled(False)
        self.StopCntBTN.setEnabled(True)
        self.isCounting.emit(True)

    def stopCounting(self):
        self.StopCntBTN.setEnabled(False)
        self.StartCntBTN.setEnabled(True)
        self.isCounting.emit(False)

    def resetCounter(self):
        self.detValues = {x:0 for x in labels}
        self.counterSet.emit(self.detValues)
        self.updateObjectsTE()
        self.LogTE.append(f"{QDateTime.currentDateTime().toString('d-MM-yyyy  hh:mm:ss')}: Licznik został zresetowany")
    def resetPopup(self):
        pop = QMessageBox()
        pop.setWindowTitle("Reset licznika")
        pop.setIcon(QMessageBox.Question)
        pop.setText("Czy na pewno chcesz zresetować licznik?")
        pop.setStandardButtons(QMessageBox.Ok|QMessageBox.Cancel)
        pop.setDefaultButton(QMessageBox.Cancel)
        ret = pop.exec_()
        if ret == QMessageBox.Ok:
            self.resetCounter()
        else:
            return

if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()