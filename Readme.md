# Face Mask Detection using OpenCV

This project demonstrates real-time face mask detection using OpenCV and a pre-trained Haar Cascade classifier.

## Installation

1. Clone the repository to your local machine:

   git clone https://github.com/uktentu/FaceMaskDetctCV.git

2. Navigate to the project directory:

   cd face-mask-detection

3. Create a virtual environment (optional but recommended):

   For Windows:
   python -m venv venv

   For macOS/Linux:
   python3 -m venv venv

4. Activate the virtual environment:

   For Windows:
   venv\Scripts\activate

   For macOS/Linux:
   source venv/bin/activate

5. Install required packages using pip:

   pip install -r requirements.txt

## Usage

1. Run the face mask detection script:

   python face_mask_detection.py

2. The program will open a window showing the live webcam feed with face mask detection. Press 'q' to exit the program.

3. Optionally, you can click the "Quit" button on the window to exit as well.

## Notes

- Ensure you have a webcam connected to your computer for live video feed.
- Adjust the threshold for mask detection (mask_percentage > 10) in the code as needed for your environment.
