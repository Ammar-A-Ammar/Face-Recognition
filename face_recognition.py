import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QSpacerItem, QSizePolicy, QMenuBar, QAction, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Face Recognition with Eigenfaces')
        self.setGeometry(100, 100, 900, 700)

        # Add menu bar for theme toggling
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.theme_menu = self.menu_bar.addMenu('Theme')
        
        self.light_mode_action = QAction('Light Mode', self, checkable=True)
        self.light_mode_action.setChecked(True)
        self.light_mode_action.triggered.connect(self.set_light_mode)
        self.theme_menu.addAction(self.light_mode_action)

        self.dark_mode_action = QAction('Dark Mode', self, checkable=True)
        self.dark_mode_action.triggered.connect(self.set_dark_mode)
        self.theme_menu.addAction(self.dark_mode_action)

        self.purple_mode_action = QAction('Purple Mode', self, checkable=True)
        self.purple_mode_action.triggered.connect(self.set_purple_mode)
        self.theme_menu.addAction(self.purple_mode_action)

        self.blue_mode_action = QAction('Blue Mode', self, checkable=True)
        self.blue_mode_action.triggered.connect(self.set_blue_mode)
        self.theme_menu.addAction(self.blue_mode_action)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Train model button
        self.train_button = QPushButton('Train', self)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setToolTip('Train the face recognition model')
        self.main_layout.addWidget(self.train_button)

        # Layout for Load Image section
        self.load_layout = QHBoxLayout()
        self.main_layout.addLayout(self.load_layout)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setToolTip('Load an image from the dataset')
        self.load_layout.addWidget(self.load_button)

        self.image_number_input = QLineEdit(self)
        self.image_number_input.setPlaceholderText("Enter image number")
        self.image_number_input.setToolTip('Enter the number of the image to load from the dataset')
        self.load_layout.addWidget(self.image_number_input)

        # Recognize button
        self.recognize_button = QPushButton('Recognize', self)
        self.recognize_button.clicked.connect(self.recognize_face)
        self.recognize_button.setToolTip('Recognize the loaded image')
        self.main_layout.addWidget(self.recognize_button)

        # Image layout with spacers
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.image_label)

        # Message layout with spacers
        self.message_layout = QVBoxLayout()
        self.main_layout.addLayout(self.message_layout)

        # Image layout with spacers
        self.super_top_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(self.super_top_spacer)

        # Add top spacer
        self.top_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.message_layout.addItem(self.top_spacer)

        # Message labels
        self.message_label = QLabel(self)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setFont(QFont('Arial', 20))
        self.main_layout.addWidget(self.message_label, alignment=Qt.AlignCenter)

        self.distance_label = QLabel(self)
        self.distance_label.setAlignment(Qt.AlignCenter)
        self.distance_label.setFont(QFont('Arial', 20))
        self.distance_label.setStyleSheet("color: red")
        self.main_layout.addWidget(self.distance_label, alignment=Qt.AlignCenter)

        self.additional_info_label = QLabel(self)
        self.additional_info_label.setAlignment(Qt.AlignCenter)
        self.additional_info_label.setFont(QFont('Arial', 20))
        self.main_layout.addWidget(self.additional_info_label, alignment=Qt.AlignCenter)

        # Add bottom spacer
        self.bottom_spacer = QSpacerItem(80, 160, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.message_layout.addItem(self.bottom_spacer)

        self.faces_image = np.load('./data_set_faces.npy')
        self.faces_target = np.load('./data_set_target.npy')

        self.mean_face = None
        self.eigen_vecs = None
        self.faces_norm = None
        self.loaded_image_index = None  # Store the loaded image index
        self.projections = None  # Store the projections of the training faces
        self.number_of_train_data = 200

    def train_model(self):
        faces_data = self.faces_image.reshape(self.faces_image.shape[0], self.faces_image.shape[1] * self.faces_image.shape[2])

        neutral = []
        for i in range(self.number_of_train_data):
            img2 = np.array(self.faces_image[i]).flatten()
            neutral.append(img2)

        faces_matrix = np.vstack(neutral)
        self.mean_face = np.mean(faces_matrix, axis=0)

        faces_norm = faces_matrix - self.mean_face
        self.faces_norm = faces_norm

        face_cov = np.cov(faces_norm.T)

        eigen_vecs, eigen_vals, _ = np.linalg.svd(face_cov)
        self.eigen_vecs = eigen_vecs

        self.projections = self.faces_norm.dot(self.eigen_vecs[:, :self.number_of_train_data])  # Project training faces
        self.display_eigenfaces()

    def load_image(self):
        image_number = self.image_number_input.text()
        if image_number.isdigit():
            image_number = int(image_number)
            if 0 <= image_number < len(self.faces_image):
                self.loaded_image_index = image_number
                loadImage = (self.faces_image[image_number] * 255).astype(np.uint8)
                height, width = loadImage.shape
                qimage = QImage(loadImage.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
                self.message_label.setText(f"Loaded image {image_number} from dataset.")
                self.message_label.setStyleSheet("color: black")
                self.distance_label.setText("")  # Clear distance label
                self.additional_info_label.setText("")  # Clear additional info label
            else:
                self.show_error_message("Image number out of range.")
        else:
            self.show_error_message("Please enter a valid number.")

    def display_eigenfaces(self):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in np.arange(10):
            ax = plt.subplot(2, 5, i + 1)
            img = self.eigen_vecs[:, i].reshape(64, 64)
            ax.imshow(img, cmap='gray')
        fig.suptitle("First 10 Eigenfaces", fontsize=16)
        plt.show()

    def recognize_face(self):
        if self.mean_face is None or self.eigen_vecs is None:
            self.show_error_message("Model not trained yet!")
            return

        if self.loaded_image_index is None:
            self.show_error_message("No image loaded to recognize.")
            return

        loadImage = self.faces_image[self.loaded_image_index]
        test_image = np.array(loadImage).flatten()

        test_image_norm = test_image - self.mean_face
        test_projection = test_image_norm.dot(self.eigen_vecs[:, :self.number_of_train_data])

        # Calculate Euclidean distances to all training projections
        distances = np.linalg.norm(self.projections - test_projection, axis=1)
        min_distance = np.min(distances)

        threshold = 7  # Set a threshold for recognition
        if min_distance < threshold:
            self.message_label.setText(f"Image {self.loaded_image_index} recognized with distance {min_distance:.2f}")
            self.message_label.setStyleSheet("color: green")
            self.distance_label.setText("")  # Clear distance label
            self.additional_info_label.setText("")  # Clear additional info label
        else:
            self.message_label.setText("Image not recognized in the dataset.")
            self.message_label.setStyleSheet("color: red")
            self.distance_label.setText(f"It's distance is {min_distance:.2f}")
            self.additional_info_label.setText(f"More than the threshold by {min_distance - threshold:.2f}")
            self.additional_info_label.setStyleSheet("color: red")

    def set_light_mode(self):
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                color: black;
            }
        """)
        self.update_button_styles("")
        self.message_label.setStyleSheet("color: black")
        self.distance_label.setStyleSheet("color: red")
        self.additional_info_label.setStyleSheet("color: black")
        self.uncheck_other_modes(self.light_mode_action)

    def set_dark_mode(self):
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: white;
            }
        """)
        self.update_button_styles("background-color: #444; color: white;", "background-color: #666;")
        self.message_label.setStyleSheet("color: white")
        self.distance_label.setStyleSheet("color: red")
        self.additional_info_label.setStyleSheet("color: white")
        self.uncheck_other_modes(self.dark_mode_action)

    def set_purple_mode(self):
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: #5E3C58;
                color: white;
            }
        """)
        self.update_button_styles("background-color: #7D5C7D; color: white;", "background-color: #9D7C9D;")
        self.message_label.setStyleSheet("color: white")
        self.distance_label.setStyleSheet("color: yellow")
        self.additional_info_label.setStyleSheet("color: white")
        self.uncheck_other_modes(self.purple_mode_action)

    def set_blue_mode(self):
        self.central_widget.setStyleSheet("""
            QWidget {
                background-color: #3A4A6A;
                color: white;
            }
        """)
        self.update_button_styles("background-color: #4A5A7A; color: white;", "background-color: #6A7A9A;")
        self.message_label.setStyleSheet("color: white")
        self.distance_label.setStyleSheet("color: cyan")
        self.additional_info_label.setStyleSheet("color: white")
        self.uncheck_other_modes(self.blue_mode_action)

    def update_button_styles(self, normal_style, hover_style=""):
        button_style = f"""
            QPushButton {{
                {normal_style}
            }}
            QPushButton:hover {{
                {hover_style}
            }}
        """
        self.train_button.setStyleSheet(button_style)
        self.load_button.setStyleSheet(button_style)
        self.recognize_button.setStyleSheet(button_style)

    def uncheck_other_modes(self, current_action):
        actions = [self.light_mode_action, self.dark_mode_action, self.purple_mode_action, self.blue_mode_action]
        for action in actions:
            if action != current_action:
                action.setChecked(False)
        current_action.setChecked(True)

    def show_error_message(self, message):
        self.message_label.setText(message)
        self.message_label.setStyleSheet("color: red")
        self.distance_label.setText("")  # Clear distance label
        self.additional_info_label.setText("")  # Clear additional info label

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = FaceRecognitionApp()
    mainWindow.show()
    sys.exit(app.exec_())
