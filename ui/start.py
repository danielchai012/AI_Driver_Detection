from PyQt5 import QtWidgets
from controller import MainWindow_ctrl
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_ctrl()
    window.show()
    sys.exit(app.exec_())
