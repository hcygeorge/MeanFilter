# pyinstaller -D -w filter_UI.py
# -D: genernate folder, -w: no console, -i: set icon
# build: store temp files, dist: store execution file
#%%
# Working directory
import os
os.chdir(os.path.join(os.getcwd(), 'C:/works/PythonCode/MeanFilter'))
os.getcwd()
import sys #, time
from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtCore import QBasicTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QProgressBar, QFileDialog, QFrame, QSpinBox
import numpy as np
from PIL import Image
from add_noise import add_gaussian_noise, add_impulse_noise
from huffman_code import img_encoder, img_decoder
#%%
class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(1000, 200, 1280, 720)  # x, y, h, w
        self.setWindowTitle("Image Filter Kit")
        self.initUI()
        self.timeLimit = 60  # todo
        self.size_filter = 7
        
        
    def initUI(self):
        self.setWindowIcon(QIcon('./data/ntpu_logo.png'))
        self.info = ['Mean\n(Arithmetic)', 'Mean\n(Geometric) ', 'Mean\n(Harmonic)', 'Contra\nHarmonic(Q=1)', 'Contra\nHarmonic(Q=-2)', 'Median']
        self.filter_abbr = ['A', 'G', 'H', 'CHQ1', 'CHQ-2', 'Median']
        self.dict_covert = dict(zip(self.info, self.filter_abbr))
        # self.setStyleSheet('QWidget{background-color:rgb(30, 30, 30)}')  # change all widgets
        # p = self.palette()
        # p.setColor(self.backgroundRole(), QtCore.Qt.black)
        # self.setPalette(p)
        # self.setStyleSheet("QWidget{background-color:rgb(30, 30, 30);color: rgb(220, 220, 220)}")
        self.setStyleSheet("QWidget{background-color:rgb(50, 50, 50);color: rgb(220, 220, 220);font-size: 20px; font-family: Arial}"
                           "QMainWindow{background-color:rgb(30, 30, 30)}"
                           "QPushButton{background-color: rgb(50, 50, 50);color: rgb(220, 220, 220);border: none}"
                           "QPushButton:hover{background-color: rgb(80, 80, 80); color: rgb(255, 255, 255)}"
                           "QLabel{background-color: rgb(50, 50, 50);color: rgb(220, 220, 220)}"
                           "QTextBrowser{background-color: rgb(30, 30, 30);color: rgb(220, 220, 220)}"
                           "QProgressBar{text-align: center;border: none}"
                        #    "QProgressBar::chunk{background-color: rgb(50, 50, 50)}"
                           "QSpinBox{background-color: rgb(50, 50, 50);border: none}"
                           "QComboBox{background-color: rgb(50, 50, 50);border: none}")

        menu_bar = self.menuBar()  # from QMainWindow
        # menu_bar.setStyleSheet("menuBar{background-color:rgb(50, 50, 50)")
        menu_file = menu_bar.addMenu('File')
        menu_noise = menu_bar.addMenu('Noise')
        menu_compress = menu_bar.addMenu('Compress')
        
        action_new = QtWidgets.QAction(self)
        action_new.setText("New image")
        action_new.setShortcut("Crtl+N")
        action_new.setObjectName("new")
        action_new.setStatusTip('New image')
        
        action_new.triggered.connect(self.get_image_file)
        
        action_huffman_encoding = QtWidgets.QAction(self)
        action_huffman_encoding.setText("Huffman Encoder")
        action_huffman_encoding.setShortcut("Crtl+H")
        action_huffman_encoding.setObjectName("huffman_encoding")
        action_huffman_encoding.setStatusTip('huffman encoding')
        action_huffman_encoding.triggered.connect(self.encoder)        

        action_huffman_decoding = QtWidgets.QAction(self)
        action_huffman_decoding.setText("Huffman Decoder")
        action_huffman_decoding.setShortcut("Crtl+D")
        action_huffman_decoding.setObjectName("huffman_decoding")
        action_huffman_decoding.setStatusTip('huffman decoding')
        action_huffman_decoding.triggered.connect(self.decoder)      

        # action_saveas = QtWidgets.QAction(self)
        # action_saveas.setText("Save image as")
        # action_saveas.setShortcut("Crtl+S")
        # action_saveas.setObjectName("saveas")
        # action_saveas.setStatusTip('Save image as')
        
        action_noise_gauss = QtWidgets.QAction(self)
        action_noise_gauss.setText("Add Gaussian noise")
        action_noise_gauss.setObjectName("noise_gauss")
        action_noise_gauss.setStatusTip("Add Gaussian noise")
        action_noise_gauss.triggered.connect(self.noise_gauss)
        
        action_noise_impulse = QtWidgets.QAction(self)
        action_noise_impulse.setText("Add impulse noise")
        action_noise_impulse.setObjectName("noise_impulse")
        action_noise_impulse.setStatusTip("Add impulse noise")
        action_noise_impulse.triggered.connect(lambda: self.noise_impulse(ratio_salt=0.5))
        
        action_noise_pepper = QtWidgets.QAction(self)
        action_noise_pepper.setText("Add pepper noise")
        action_noise_pepper.setObjectName("noise_pepper")
        action_noise_pepper.setStatusTip("Add pepper noise")
        action_noise_pepper.triggered.connect(lambda: self.noise_impulse(ratio_salt=0))
        
        # action_mean_a = QtWidgets.QAction(self)
        # action_mean_a.setText("Arithmetic Mean filter")
        # action_mean_a.setObjectName("mean_a")
        # action_mean_a.setStatusTip("Arithmetic Mean filter")
        # action_mean_a.triggered.connect(self.filter_mean)
        
        # action_median = QtWidgets.QAction(self)
        # action_median.setText("Median filter")
        # action_median.setObjectName("median")
        # action_median.setStatusTip("Median filter")
        # action_median.triggered.connect(self.filter_median)
        
        menu_file.addAction(action_new)
        # menu_file.addAction(action_saveas)
        menu_noise.addAction(action_noise_gauss)
        menu_noise.addAction(action_noise_impulse)
        menu_noise.addAction(action_noise_pepper)
        menu_compress.addAction(action_huffman_encoding)
        menu_compress.addAction(action_huffman_decoding)
        # menu_filter.addAction(action_mean_a)
        # menu_filter.addAction(action_median)
        
        # Image on the left
        self.label_img_left = QtWidgets.QLabel(self)
        self.label_img_left.setGeometry(QtCore.QRect(100, 100, 480, 480))
        self.label_img_left.setFrameShape(QFrame.Panel)
        self.label_img_left.setFrameShadow(QFrame.Sunken)
        self.label_img_left.setPixmap(QPixmap("./data/pic.png"))
        self.label_img_left.setScaledContents(True)  # scaling image to fit the label size
        self.label_img_left.setObjectName("label_img_left")
        
        # Image on the right
        self.label_img_right = QtWidgets.QLabel(self)
        self.label_img_right.setGeometry(QtCore.QRect(700, 100, 480, 480))
        self.label_img_right.setFrameShape(QFrame.Panel)
        self.label_img_right.setFrameShadow(QFrame.Sunken)
        self.label_img_right.setPixmap(QPixmap("./data/pic.png"))
        self.label_img_right.setScaledContents(True)
        self.label_img_right.setObjectName("label_img_right")
        
        # Info
        self.editor_left = QtWidgets.QTextBrowser(self)
        self.editor_left.setGeometry(QtCore.QRect(100, 600, 480, 80))
        self.editor_left.setText("Please upload an image or use the sample image...")
        self.editor_left.setObjectName("editor_left")
        
        # Editor of filter size
        self.spinSize = QtWidgets.QSpinBox(self)
        self.spinSize.setGeometry(QtCore.QRect(820, 40, 120, 50))
        self.spinSize.setRange(3, 111)
        self.spinSize.setValue(3)
        self.spinSize.setSingleStep(2)
        self.spinSize.setPrefix("Size: ")
        self.spinSize.setWrapping(True)
        self.spinSize.setObjectName("spinSize")
        
        # Progress bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(QtCore.QRect(700, 640, 480, 40))
        self.progress.setMaximum(20)
        
        # Button
        self.upload = QtWidgets.QPushButton(self)
        self.upload.setGeometry(QtCore.QRect(100, 40, 120, 50))
        self.upload.setObjectName('upload_image')
        self.upload.setText('New')
        self.upload.setIcon(QIcon('./data/add.png'))
        self.upload.setIconSize(QtCore.QSize(36, 36))
        self.upload.clicked.connect(self.get_image_file)

        self.sample01 = QtWidgets.QPushButton(self)
        self.sample01.setGeometry(QtCore.QRect(220, 40, 120, 50))
        self.sample01.setObjectName("sample01")
        self.sample01.setText('Sample')
        self.sample01.setIcon(QIcon('./data/sample.png'))
        self.sample01.setIconSize(QtCore.QSize(36, 36))
        
        self.undo = QtWidgets.QPushButton(self)
        self.undo.setGeometry(QtCore.QRect(340, 40, 120, 50))
        self.undo.setText('Undo')
        self.undo.setObjectName("undo")
        self.undo.setIcon(QIcon('./data/undo.png'))
        self.undo.setIconSize(QtCore.QSize(36, 36))
        
        self.save_left = QtWidgets.QPushButton(self)
        self.save_left.setGeometry(QtCore.QRect(460, 40, 120, 50))
        self.save_left.setText('Save')
        self.save_left.setObjectName("Save")
        self.save_left.setIcon(QIcon('./data/save.png'))
        self.save_left.setIconSize(QtCore.QSize(34, 34))
        self.save_left.clicked.connect(lambda: self.imageSaveAs('l'))
        
        self.saveR = QtWidgets.QPushButton(self)
        self.saveR.setGeometry(QtCore.QRect(1060, 40, 120, 50))
        self.saveR.setText('Save')
        self.saveR.setObjectName("saveR")
        self.saveR.setIcon(QIcon('./data/save.png'))
        self.saveR.setIconSize(QtCore.QSize(34, 34))
        self.saveR.clicked.connect(lambda: self.imageSaveAs('r'))
        
        self.comboFilter = QtWidgets.QComboBox(self)
        self.comboFilter.setGeometry(QtCore.QRect(700, 40, 120, 50))
        self.comboFilter.setObjectName("Filter")
        self.spinSize.setPrefix("Size: ")
        self.comboFilter.addItems(self.info)
        
        self.buttonFilter = QtWidgets.QPushButton(self)
        self.buttonFilter.setGeometry(QtCore.QRect(940, 40, 120, 50))
        self.buttonFilter.setText('Apply')
        self.buttonFilter.setObjectName("Apply")
        self.buttonFilter.setIcon(QIcon('./data/filter.png'))
        self.buttonFilter.setIconSize(QtCore.QSize(34, 34))
        self.buttonFilter.clicked.connect(self.applyFilter)        
        
        self.labelStatus = QtWidgets.QLabel(self)
        self.labelStatus.setGeometry(QtCore.QRect(700, 600, 480, 30))
        self.labelStatus.setObjectName('labelStatus')
        self.labelStatus.setText('Standby.')
        self.labelStatus.setStyleSheet("QLabel{background-color:rgb(30, 30, 30)}")

        
        # Trigger
        QtCore.QMetaObject.connectSlotsByName(self)
        self.undo.clicked.connect(self.undo_image)
        self.sample01.clicked.connect(self.load_sample)

        
    # def retranslateUi(self):
    #     _translate = QtCore.QCoreApplication.translate
    #     self.setWindowTitle(_translate("MainWindow", "MainWindow"))
    #     self.cat.setText(_translate("MainWindow", "Cat"))
    #     self.dog.setText(_translate("MainWindow", "Dog"))
    
    def display_info(self, info):
        ''' Display the info in the texteditor and scroll down.
        '''
        self.editor_left.append(str(info))
        self.editor_left.verticalScrollBar().setValue(self.editor_left.verticalScrollBar().maximum())


    def get_image_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open image file", ".",
                                                       'Image files (*.jpg *.jpeg *.png)')
        self.img_origin = np.array(Image.open(self.filename).convert('L'))  # hold original image
        self.img_noise = self.img_origin.copy()
        self.img_shape = self.img_noise.shape
        self.label_img_left.setPixmap(QPixmap(self.filename))  # display image on the left
        self.label_img_left.setToolTip(str(self.filename))
        self.display_info("Upload Image")
        self.display_info("-> Image Shape:{}".format(str(self.img_noise.shape)))
        
    def imageSaveAs(self, side):
        filename, _ = QFileDialog.getSaveFileName(self, "Save image as", ".", 'Image files (*.jpg *.jpeg *.png)')
        if side == 'l':
            img = Image.fromarray(self.img_noise)
        else:
            img = Image.fromarray(self.img_right)
        img.save(filename)
            
        # file = open(filename, 'w')
        # image = self.img_noise
        # h, w = image.shape
        # qimage = QImage(image.data, w, h, w,
        #                       QImage.Format_Grayscale8) 
        # file.write(qimage)
        # file.close()
    
    def load_sample(self):
        self.filename = "./data/jordan.jpg"
        self.img_origin = np.array(Image.open(self.filename).convert('L'))  # keep original image
        self.img_noise = self.img_origin.copy()  # image to process
        self.img_shape = self.img_noise.shape
        self.label_img_left.setPixmap(QPixmap(self.filename))
        self.label_img_left.setToolTip(str(self.filename))
        self.display_info("-> Load sample image 'jordan.jpg'")
        self.display_info("-> Image Shape:{}".format(str(self.img_shape)))
                
    def undo_image(self):
        self.img_noise = self.img_origin.copy()
        self.label_img_left.setPixmap(QPixmap(self.filename))
        self.display_info("-> Recover image")
        
    def display_image(self, image, qlabel):
        '''Convert ndarray to QImage and display on the label.
        
        Used in filter and noise function.
        
        '''
        h, w = image.shape
        qimage = QImage(image.data, w, h, w,
                              QImage.Format_Grayscale8)  # covert ndarray to QImage
        qlabel.setPixmap(QPixmap(qimage))
        
    def encoder(self):
        self.encoded, self.node, self.code, self.freq = img_encoder(self.img_noise)
        self.display_info("-> Huffman Encoding")
        self.display_info("-> Original Size:{} bits".format(self.img_noise.size*8))
        self.display_info("-> Encoding Size:{} bits".format(len(self.encoded)))
        self.display_info("-> Compression Rate:{:.2f}%".format(100*len(self.encoded) / (self.img_noise.size*8))) 
        self.display_info("-> Do NOT close the program if you want to decode the image.")

        filename, _ = QFileDialog.getSaveFileName(self, "Save Compressed File as txt", ".", "Text Files (*.txt)")
        with open(filename, 'w') as f:
            f.write(self.encoded)
        # filename, _ = QFileDialog.getSaveFileName(self, "Save Encoding list as txt",".", "Text Files (*.txt)")
        # with open(filename, 'w') as f:
        #     f.write(str(self.code))
        # filename, _ = QFileDialog.getSaveFileName(self, "Save freq table as txt", "Text Files (*.txt)")
        # with open(filename, 'w') as f:
        #     f.write(str(self.freq))
            
    def decoder(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Encoding File", ".", "Text Files (*.txt)")
        with open(filename, 'r') as f:
            self.encoded_load = f.readlines()[0]
        self.display_info("-> Encoding Size:{} bits".format(len(self.encoded_load)))
        self.display_info("-> Huffman Decoding")
        self.decoded = img_decoder(self.encoded_load, self.node, self.code, self.img_shape)
        self.display_image(self.decoded.astype('uint8'), self.label_img_right)
        
    def noise_gauss(self):
        self.img_noise = add_gaussian_noise(self.img_noise)
        self.display_image(self.img_noise, self.label_img_left)
        self.display_info("-> add Gaussian noise")
        
    def noise_impulse(self, ratio_salt):
        self.img_noise = add_impulse_noise(self.img_noise, 0.05, ratio_salt)
        self.display_image(self.img_noise, self.label_img_left)
        self.display_info("-> add impulse noise")
        
    def applyFilter(self):
        methods = self.dict_covert[self.comboFilter.currentText()]
        size = int(self.spinSize.value())
        filters = ImageFilter()
        if methods == 'Median':
            self.labelStatus.setText('Progressing: Median filter')
            self.img_right = filters.filter_median(self.img_noise, size, self.progress)
            self.display_image(self.img_right, self.label_img_right)
            self.display_info("-> apply median filter %dx%d" % (size, size))
        elif methods == 'CHQ1':
            self.labelStatus.setText('Progressing: Contraharmonic mean filter')
            self.img_right = filters.filter_contraharmonic(self.img_noise, size, self.progress, Q=1)
            self.display_image(self.img_right, self.label_img_right)
            self.display_info("-> apply contraharmonic mean filter %dx%d" % (size, size))
        elif methods == 'CHQ-2':
            self.labelStatus.setText('Progressing: Contraharmonic mean filter')
            self.img_right = filters.filter_contraharmonic(self.img_noise, size, self.progress, Q=-2)
            self.display_image(self.img_right, self.label_img_right)
            self.display_info("-> apply contraharmonic mean filter %dx%d" % (size, size))
        else:
            self.labelStatus.setText('Progressing: Mean filter')
            self.img_right = filters.filter_mean(self.img_noise, size, self.progress, methods)
            self.display_image(self.img_right, self.label_img_right)
            self.display_info("-> apply mean filter %s %dx%d" % (methods, size, size))
            
        self.labelStatus.setText('Done.')




#%%
# Class filter
class ImageFilter():
    def __init__(self):
        self.filter_size = 3
        self.filters_dict = {}
        
    def filter_mean(self, image, filter_size, progress_bar, method):
        '''One channel images mean filter'''
        out = np.copy(image)  # create a new copy of image
        print((out == image).all())
        count = 0
        progress_bar.setMaximum(out.shape[0] - filter_size + 1)  # self.progress.setMaximum
        for row in range(out.shape[0] - filter_size + 1):
            for col in range(out.shape[1] - filter_size + 1):
                block = image[row:row + filter_size, col:col + filter_size]  # why use out will makes G and H wrong?
                if method == 'A':
                    f_hat = np.sum(block) / filter_size**2  # arithmetic mean
                elif method == 'G':
                    f_hat = np.exp(np.sum(np.log(block + 1e-10)) / filter_size**2)  # geometric mean
                    # f_hat = np.prod(block)**(1 / filter_size**2)
                elif method == 'H':
                    f_hat = filter_size**2 / np.sum(1/(block + 1e-10))  # harmonic mean

                out[row + filter_size//2, col + filter_size//2] = f_hat
            count += 1
            progress_bar.setValue(count)
        return out
    
    def filter_median(self, image, filter_size, progress_bar):
        out = np.copy(image)  # create a new copy of image
        count = 0
        progress_bar.setMaximum(out.shape[0] - filter_size + 1)  # self.progress.setMaximum
        
        for row in range(out.shape[0] - filter_size + 1):
            for col in range(out.shape[1] - filter_size + 1):
                block = out[row:row + filter_size, col:col + filter_size]
                f_hat = np.median(block.reshape(1, -1))  # median
                out[row + filter_size//2, col + filter_size//2] = f_hat
            count += 1    
            progress_bar.setValue(count)  # self.progress.setValue
        return out
    
    def filter_contraharmonic(self, image, filter_size, progress_bar, Q):
        out = np.copy(image).astype('float64')  # uint can't do power operation! 
        count = 0
        progress_bar.setMaximum(out.shape[0] - filter_size + 1)  # self.progress.setMaximum
        
        for row in range(out.shape[0] - filter_size + 1):
            for col in range(out.shape[1] - filter_size + 1):
                block = image[row:row + filter_size, col:col + filter_size]  # why?
                f_hat = np.sum(np.float_power(block, Q + 1)) / (np.sum(np.float_power(block, Q)) + 1e-10)
                out[row + filter_size//2, col + filter_size//2] = np.round(f_hat)
            count += 1    
            progress_bar.setValue(count)  # self.progress.setValue
        return out.astype('uint8')
    

   
# if __name__ == "__main__":
def window():
    app = QtWidgets.QApplication(sys.argv)
    ui = UI()  # self defined window
    ui.show()
    sys.exit(app.exec_())

window()


#%%
# with open('C:/Users/user/Desktop/123.txt', 'r') as f:
#     encoded = f.readlines()

# len(encoded[0])