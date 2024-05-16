import windowUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
#import pandas
import os
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
#import math
import cv2
import keras
from PIL import Image
import time
import psutil
'''''
 #--------------------------plotWidget------------------
        self.plotWidget = QtWidgets.QWidget(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self.plotWidget)
        #self.plotWidget.setGeometry(QtCore.QRect(0, 720, 471, 261))#x,y,w,h
        self.plotWidget.setStyleSheet("")
        self.plotWidget.setObjectName("plotWidget")
        self.plotLayout = QtWidgets.QVBoxLayout(self.plotWidget)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)  
        self.plotLayout.setObjectName("plotLayout")
        #-------------------------Connection-----------------------
        
        logic.giveWindow(self, MainWindow)
        self.actionOpenFile.triggered.connect(logic.OpenFile_fun)
        self.actionTakeFrame.triggered.connect(logic.TakeFrame_fun)
        self.actionZoomIn.triggered.connect(logic.ZoomIn_fun)
        self.actionZoomOut.triggered.connect(logic.ZoomOut_fun)

'''

window = "window I have"
Mainwindow = "Strange thing"
model = "model"
def giveWindow(se_lf, mainWin):
    global window
    global Mainwindow
    window = se_lf
    Mainwindow = mainWin
    #need to do once
    window.fig = Figure(facecolor= (0.572, 0.65, 0.541))
    window.ax = window.fig.add_subplot()
    window.canvas = FigureCanvas(window.fig)
    window.plotLayout.addWidget(window.canvas)

    cid = window.fig.canvas.mpl_connect('button_press_event', onclick)
    global model  
    model = ""


img = "loaded image"
frameSize =50

def OpenFile_fun():
    print("opening file here")
    modelPath = QFileDialog.getOpenFileName(Mainwindow, filter="*.h5")[0]
    global model  
    model = keras.models.load_model(modelPath)

    fileNm = QFileDialog.getOpenFileName(Mainwindow, filter="Image Files (*.png *.jpg *.bmp);; Video Files (*.mp4 *.avi *.gif *.webm)")[0]
    
    file_extension = os.path.splitext(fileNm)[1]
    print(file_extension)
    if(file_extension == ".mp4"):
        print("video")
        VideoProcesssing_YOLOv8(fileNm)
    else:
        if(len(fileNm) > 0):
            global img
            img = cv2.imread(fileNm)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img.shape)
            window.ax.imshow(img)
            window.canvas.draw()




def ZoomIn_fun():
    print("ZoomIn")
    global img
    if(type(img) != type("string")):
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        print(f"resize  {img.shape}")
        window.ax.imshow(img)
        window.canvas.draw()        
    str_for_label = f"\nImage size : {img.shape}"
    window.label.setText(str_for_label)

def ZoomOut_fun():
    print("ZoomOut")
    global img
    if(type(img) != type("string")):      

        img = cv2.resize(img, (img.shape[0] // 2 if img.shape[0] // 2 >= frameSize else frameSize,
                                    img.shape[1] // 2 if img.shape[1] // 2 >= frameSize else frameSize))
        
        print(f"\t\t-----Resize  {img.shape}-----")
        window.ax.imshow(img)
        window.canvas.draw()
       
    str_for_label = f"\nImage size : {img.shape}"
    window.label.setText(str_for_label)

ix, iy = 0, 0
time_start = 0
def onclick(event):
    global ix, iy, time_start
    time_start = time.time()
    ix, iy = event.xdata, event.ydata
    print (f'True Center:\tx = {ix}, y = {iy} frameSize = {frameSize}')
    # shift center to fit the square to image borders
    if event.xdata + frameSize/2 > img.shape[1]: # [0]- h, [1]- w
        ix = img.shape[1] - frameSize/2
    if event.xdata - frameSize/2 < 0:
        ix = 0 + frameSize/2
    #else:
    #    ix = event.xdata

    if event.ydata + frameSize/2 > img.shape[0]:
        iy = img.shape[0] - frameSize/2
    if event.ydata - frameSize/2 < 0:
        iy = 0 + frameSize/2
    #else:
    #    iy = event.ydata
    drawSquareAndPred()

time_finish = 0
def drawSquareAndPred():
    print (f'Shifted Center:\tx = {ix}, y = {iy} frameSize = {frameSize}')
    frame_img = img.copy()

    print ("----------------------")
    left_TOP_x = int(ix - frameSize/2)
    left_TOP_y = int(iy - frameSize/2)
    right_BOTTOM_x = int(ix + frameSize/2)
    right_BOTTOM_y = int(iy + frameSize/2)
    print (f'to int\t left_TOP_x = {left_TOP_x}, left_TOP_y = {left_TOP_y}')
    print (f'to int\t right_BOTTOM_x = {right_BOTTOM_x}, right_BOTTOM_y = {right_BOTTOM_y}')
    print ("----------------------\n")

    cv2.rectangle(frame_img, (left_TOP_x,left_TOP_y), (right_BOTTOM_x, right_BOTTOM_y ), color=(255, 0, 0), thickness= 2)
    window.ax.imshow(frame_img)
    window.canvas.draw()
    crop_img = img[left_TOP_y:right_BOTTOM_y, left_TOP_x:right_BOTTOM_x]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    #print(f"SIZE\t{len(crop_img)}\t {len(crop_img[0])}")
    cv2.imshow("imagggge", crop_img)
    test_image(crop_img)
    global time_finish
    time_finish = time.time()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   #window.fig.canvas.mpl_disconnect(cid)

#metal, glass, biological, paper, battery, trash, cardboard, shoes, clothes, and plastic
predictions = {0: 'metal',
               1: 'glass', 
               2: 'biological', 
               3: 'paper', 
               4: 'battery',
               5: 'trash', 
               6: 'cardboard', 
               7: 'shoes', 
               8: 'clothes', 
               9: 'plastic'}
   
str_for_label = ""
def test_image(predict_image):
 
  matrix = predict_image 
  print(f"matrix.shape = {matrix.shape}")
  matrix = matrix.reshape((1, 50, 50, 3)) 
  classifications = model.predict(matrix) 
  print(classifications.shape)
  print(classifications[0])
  predicted_class_index=numpy.argmax(classifications,axis=1)
  print(f"INDEX: {predicted_class_index}")
  print(f"This is\t{predictions[predicted_class_index[0]]}\n")
  global   str_for_label
  str_for_label = ""
  str_for_label += f"INDEX: {predicted_class_index}\t {classifications[0][predicted_class_index[0]]}\n"
  str_for_label += f"This is\t{predictions[predicted_class_index[0]]}\n------------------------------------------------\n\n"
  #2nd biggest
  classifications[0][predicted_class_index[0]] = 0
  predicted_class_index=numpy.argmax(classifications,axis=1)
  print(f"INDEX: {predicted_class_index}")
  print(f"OR this is\t{predictions[predicted_class_index[0]]}\n")


  str_for_label += f"INDEX: {predicted_class_index}\t {classifications[0][predicted_class_index[0]]}\n"
  str_for_label += f"This is\t{predictions[predicted_class_index[0]]}\n------------------------------------------------\n\n"
  
  #3thd biggest
  classifications[0][predicted_class_index[0]] = 0
  predicted_class_index=numpy.argmax(classifications,axis=1)
  print(f"INDEX: {predicted_class_index}")
  print(f"OR this is\t{predictions[predicted_class_index[0]]}\n")

  str_for_label += f"INDEX: {predicted_class_index}\t {classifications[0][predicted_class_index[0]]}\n"
  str_for_label += f"This is\t{predictions[predicted_class_index[0]]}\n------------------------------------------------\n"
  
  str_for_label +=  f"\nImage size : {img.shape}"
  deltaTime = str(time.time() - time_start) # need str(...)!
  str_for_label +=  f"\nExecuting time: {deltaTime}" 
  str_for_label +=  f"\n {cpu_usage(deltaTime=deltaTime)}"
  
  window.label.setText(str_for_label)
  #detPrint(str_for_label, f"\nImage size : {img.shape}")

  def detPrint(predStr:str, sizeStr:str):
       window.label.setText(predStr + sizeStr)

def VideoProcesssing_YOLOv8(filename, minConf = 0.1, realtime = False):
    pass

def cpu_usage(deltaTime:float):
    return  f"\n'The delta CPU usage is: {psutil.cpu_percent(float(deltaTime))}"









if __name__ == "__main__":
    print("entered")
    import sys
    app = QtWidgets.QApplication(sys.argv)
    print("175")
    MainWindow = QtWidgets.QMainWindow()
    print("177")
    ui = windowUI.Ui_MainWindow()
    print("179")
    ui.setupUi(MainWindow)
    print("181")
    MainWindow.show()
    print("183")
    sys.exit(app.exec_())