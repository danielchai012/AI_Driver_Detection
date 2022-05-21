
from PyQt5.QtCore import QThread, pyqtSignal
from UI import mainMenu_UI
from PyQt5 import QtWidgets, QtGui, QtCore

from importlib.resources import path
import os
import time
import sys


#import face_recon
sys.path.append(os.getcwd())
absolutepath = os.path.abspath(__file__)
print(absolutepath)

fileDirectory = os.path.dirname(__file__)
print(fileDirectory)
# Path of parent directory
parentDirectory = os.path.dirname(os.path.dirname(__file__))
print(parentDirectory)
# Navigate to Strings directory
newPath = os.path.join(parentDirectory, 'face_recon')
print(newPath)
sys.path.append(newPath)
# print(os.getcwd())
print(sys.path)


class MainWindow_ctrl(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow_ctrl, self).__init__()
        self.ui = mainMenu_UI()
        self.ui.setupUi(self)
        self.ui.start_Button.clicked.connect(self.start_buttonClicked)
        self.ui.quit_Button.clicked.connect(self.quit_buttonClicked)

        # self.setup_control()

    def start_buttonClicked(self):
        print("You clicked start button.")
        self.ui.start_Button.setEnabled(False)
        self.qthread = ThreadTask()
        # self.qthread.qthread_signal.connect(self.progress_start)
        self.close()
        self.qthread.start_progress()

    def quit_buttonClicked(self):
        sys.exit(QtWidgets.QApplication(sys.argv))

    # def setup_control(self):
        # TODO
        #self.ui.textEdit.setText('Happy World!')


class ThreadTask(QThread):
    #qthread_signal = pyqtSignal(int)

    def start_progress(self):

        from face_recon import recognize_faces_video
        from face_recon import line_notify
        personlist = recognize_faces_video.start_recon()
        # message = "駕駛者是"+personlist[0]
        # line_notify.lineNotifyMessage(message)
        import headeyes_hog
        #os.system('python face_recon_dlib.py')
        # face_recon_dlib.face_recon()
        # self.qthread_signal.emit()print(sys.path)
