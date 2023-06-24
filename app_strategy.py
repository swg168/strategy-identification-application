import os
import time
import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from torchvision import transforms
import pandas as pd
import math

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap,QIcon,QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QProgressBar,
    QTextEdit,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QToolBar,
    QMessageBox
    
)


class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock, self).__init__()
        self.channel=channel
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel),
        )

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out+=x
        out=F.relu(out)
        return out
    
    
class Resnet(LightningModule):
    def __init__(self):
        super().__init__()

        # Define the model
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5), #(3,64,64)
            nn.BatchNorm2d(32),                                     #(16,60,60)
            nn.ReLU(),
            nn.MaxPool2d(2)                                         #(16,30,30)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5), #(32,26,26)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                                           #(32,13,13)
        )

        self.reslayer1=ResidualBlock(32)
        self.reslayer2=ResidualBlock(16)
        self.fc1=nn.Linear(16*29*29,2)         

       
                                
    def forward(self,x):
        out=self.conv1(x)
        out=self.reslayer1(out)
        out=self.conv2(out)
        out=self.reslayer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=torch.sigmoid(out)
        return  out


model = Resnet()
picture=[]
predict=[]

# Application of model files that are initialized loaded
check_path = f".\model_file\ResNet-epoch=84-val_acc=0.91.ckpt"


class ImageLoader(QThread):
    progress_update = pyqtSignal(int)
    image_loaded = pyqtSignal(str)

    def __init__(self, folder_path,check_path):
        super().__init__()
        self.folder_path = folder_path
        self.model = model.load_from_checkpoint(check_path,map_location=torch.device('cpu'))
        self.transform = transforms.Compose([transforms.Resize([128,128]),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            # transforms.Normalize([0.93358505, 0.9614856, 0.8578263], [0.13661127, 0.097352155, 0.24198921])
                                            ])

    def run(self):
        files = [
            f
            for f in os.listdir(self.folder_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        total_num_files = len(files)
        progress = 0
        
        for file_name in files:
        # processing image
            image_path = os.path.join(self.folder_path, file_name)
            # open the image
            image = Image.open(image_path)
            # check if it is already RGB and convert it if not
            if image.mode != "RGB":
                image = image.convert("RGB")
            else:
                image = image
            # apply the necessary transformations

            image = self.transform(image)
            image = image.view(1, 3, 128, 128)
            # if GPU
            # image=image.to('cuda')
            # load the model
            model = self.model
            # model.eval()

            # make the prediction
            with torch.no_grad():
                y_hat = model(image)

            # check the results
            if y_hat.argmax(1).cpu().numpy() == 0:
                y_label = "constructive matching"
                picture.append(file_name)
                predict.append(y_label)
            else:
                y_label = "response elimination"
                picture.append(file_name)
                predict.append(y_label)

            # update progress bar
            progress += 1
            self.progress_update.emit(int(progress / total_num_files * 100))

            # Send signal notification that the new picture has been loaded
            self.image_loaded.emit(file_name)
            

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.check_path = check_path
        # Sets the main window title
        self.setWindowTitle("Strategy Identification")
        # Set the main window size
        self.setGeometry(500, 200, 600, 400)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # Add a button component
        self.output_label = QLabel("Result", self)
        self.output_box = QTextEdit(self)
        self.btn = QPushButton("Select", self)
        # self.btn.move(50, 50)
        self.btn.clicked.connect(self.select_folder)
        
        toolbar=QToolBar()
        self.addToolBar(toolbar)
        # Create a Select Model toolbar
        select_icon=QIcon("./icon/select.svg")
        button1 = QAction(select_icon,"Select the model", self)
        button1.triggered.connect(self.select_model)
        toolbar.addAction(button1)

        # Create a Save Results toolbar
        save_icon=QIcon("./icon/save.svg")
        button2 = QAction(save_icon,"Save the results", self)
        button2.triggered.connect(self.save_result)
        toolbar.addAction(button2)
        
        # Add a progress bar component
        self.progress_bar = QProgressBar(self)

        # Add a label component
        self.label = QLabel(self)
        # self.label.setGeometry(50, 150, 500, 200)
        self.image_label = QLabel(self)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.output_box)
        hlayout.addWidget(self.image_label)
        layout = QGridLayout()
        layout.addWidget(self.btn, 0, 0)
        layout.addWidget(self.output_label, 1, 0)
        layout.addLayout(hlayout, 2, 0)
        layout.addWidget(self.label, 3, 0)
        layout.addWidget(self.progress_bar, 4, 0)
        main_widget.setLayout(layout)
        self.show()
        # Create an Image Loader instance and connect the signal slot
        self.image_loaded_count = 0
            
    def clear_picture_and_predict(self):
        global picture, predict
        picture = []
        predict = []

    def select_folder(self):
        # A folder selection box pops up
        folder_path = QFileDialog.getExistingDirectory(self, "Please choose the folder containing the eye-tracking images.", "/")
        
        # Iterate through all the pictures in the folder and update the progress bar and labels
        if folder_path:
            self.clear_picture_and_predict()
            self.output_box.clear()
            
            self.image_loader = ImageLoader(folder_path,check_path=self.check_path)
            self.image_loader.progress_update.connect(self.update_progress_bar)
            self.image_loader.image_loaded.connect(self.add_image_to_label)
            self.start_time = time.time()
            self.image_loader.start()
            self.total_num_files = len(
                [
                    f
                    for f in os.listdir(folder_path)
                    if f.endswith(".jpg") or f.endswith(".png")
                ]
            )
            self.image_loaded_count = 0
    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)
        output_text = ""
        for i in range(self.image_loaded_count, math.ceil(progress*self.total_num_files/100)):
            filename = picture[i].split(".")[0]  # Gets the file name
            output_text += f"strategy of {filename} is {predict[i]} \n"
        self.output_box.append(output_text)
        if math.ceil(progress*self.total_num_files/100) == self.total_num_files:  # When processing is complete...
            end_time = time.time()  # Record the time at which the loop ends
            elapsed_time = end_time - self.start_time  # Calculate the time spent
            self.output_box.append(
                f"\n{math.ceil(progress*self.total_num_files/100)} eye movement images were identified,\ntime-consuming：{elapsed_time:.2f} s"
            )  # Add time information to the QTextEdit
            
    def add_image_to_label(self, file_name):
        self.image_loaded_count += 1
        pixmap = QPixmap(os.path.join(self.image_loader.folder_path, file_name))
        pixmap = pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)  # Resize the picture
        self.image_label.setPixmap(pixmap)
        self.label.setText(file_name)
            
    # Create a select model function
    def select_model(self):
        file_dialog = QFileDialog(self, "Select the model file", "", "Model files (*.ckpt)")
        if file_dialog.exec():
            self.check_path = file_dialog.selectedFiles()[0]
            

    # Create a function that saves the result
    def save_result(self):
        file_dialog = QFileDialog(self, "Save the results", "", "output files (*.csv)")
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if file_dialog.exec():
            name = file_dialog.selectedFiles()[0]
            # Save the results using the file name
            df=pd.DataFrame({'picture':picture,'predict':predict} )
            df.to_csv(name,index=False)
            # Show "Save Successful" message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Save Successful")
            msg.setWindowTitle("Success")
            msg.exec()

if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon("./icon/eye.svg"))
    app.setStyle("Fusion")
    window = MyMainWindow()
    window.show()
    app.exec()
