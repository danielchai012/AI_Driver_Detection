# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainmenu_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class mainMenu_UI(object):
    def setupUi(self, mainMenu_UI):
        mainMenu_UI.setObjectName("mainMenu_UI")
        mainMenu_UI.resize(600, 800)
        mainMenu_UI.setMinimumSize(QtCore.QSize(400, 400))
        mainMenu_UI.setMaximumSize(QtCore.QSize(400, 400))
        self.label = QtWidgets.QLabel(mainMenu_UI)
        self.label.setGeometry(QtCore.QRect(0, 0, 601, 361))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(
            "../img.png"))
        self.label.setScaledContents(True)
        self.label.setWordWrap(False)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.start_Button = QtWidgets.QPushButton(mainMenu_UI)
        self.start_Button.setGeometry(QtCore.QRect(140, 250, 131, 41))
        self.start_Button.setObjectName("start_Button")
        self.quit_Button = QtWidgets.QPushButton(mainMenu_UI)
        self.quit_Button.setGeometry(QtCore.QRect(140, 300, 131, 41))
        self.quit_Button.setObjectName("quit_Button")
        self.admin_Button = QtWidgets.QPushButton(mainMenu_UI)
        self.admin_Button.setGeometry(QtCore.QRect(140, 350, 131, 41))
        self.admin_Button.setObjectName("admin_Button")

        self.retranslateUi(mainMenu_UI)
        QtCore.QMetaObject.connectSlotsByName(mainMenu_UI)

    def retranslateUi(self, mainMenu_UI):
        _translate = QtCore.QCoreApplication.translate
        mainMenu_UI.setWindowTitle(_translate("mainMenu_UI", "AI駕駛者辨識系統"))
        mainMenu_UI.setToolTip(_translate(
            "mainMenu_UI", "<html><head/><body><p><br/></p></body></html>"))
        self.start_Button.setText(_translate("mainMenu_UI", "開始"))
        self.quit_Button.setText(_translate("mainMenu_UI", "退出"))
        self.admin_Button.setText(_translate("mainMenu_UI", "管理"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
