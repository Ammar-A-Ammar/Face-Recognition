# Face Recognition with Eigenfaces

## Introduction
This project implements a face recognition system using Eigenfaces. It provides a graphical user interface built with PyQt5.

## Requirements
- Python 3
- PyQt5
- NumPy
- matplotlib
- Pillow

## Installation
You can install the required packages using pip:
```bash
pip install PyQt5 numpy matplotlib Pillow
```

## Usage

1. Clone the repository or download the files.
2. Run the `face_recognition.py` file using Python.

``` bash
python face_recognition.py
```

3. Use the menu bar to switch between different themes (Light, Dark, Purple, and Blue).
4. Click on the "Train" button to train the face recognition model.
5. Click on the "Load Image" button to load an image from the dataset.
6. Click on the "Recognize" button to recognize the loaded image.

## Files

* `face_recognition.py`: The main Python script containing the GUI implementation.
* `data_set_faces.npy`: Numpy array containing face images for training.
* `data_set_target.npy`: Numpy array containing target labels for the training images.

## License

This project is licensed under the [MIT License](LICENSE).