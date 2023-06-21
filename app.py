from flask import Flask, request
from flask_cors import CORS
import face_recognition
import base64
import numpy as np
import cv2
import os
import dlib
import math
from scipy.spatial import distance as dist
import joblib
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)



def compare_faces(reference_image_base64, target_directory):
    # Decode the base64 string to obtain the reference image
    reference_image = base64.b64decode(reference_image_base64.split(',')[1])
    nparr = np.frombuffer(reference_image, np.uint8)
    reference = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Compute the face encoding for the reference image
    encodings = face_recognition.face_encodings(reference)
    
    if len(encodings) == 0:
        return "Zero"
    if len(encodings) > 1:
        return "Multiple"
        
    reference_encoding = encodings[0]
    
    # Get a list of image files from the target directory
    target_image_paths = []
    for filename in os.listdir(target_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            target_image_paths.append(os.path.join(target_directory, filename))

    # Iterate over target images
    for target_image in target_image_paths:
        # Load the target image
        target = face_recognition.load_image_file(target_image)
        target_encoding = face_recognition.face_encodings(target)

        # Check if at least one face is found in the target image
        if len(target_encoding) > 0:
            # Compare the face in the target image with the reference face
            face_distance = face_recognition.face_distance([reference_encoding], target_encoding[0])
            threshold = 0.6
            is_same_person = face_distance[0] <= threshold

            # Return the first matching target image
            if is_same_person:
                return GetFileName(target_image)

    # Return None if no match is found
    return "None"


def GetFileName(fullPath):
    lastIndex = fullPath.rfind('\\')
    fileName = fullPath[(lastIndex+1) : ]
    return fileName


@app.route('/process_frame', methods=['POST'])
def process_frame():

    req_data = request.get_json(force=True)
    
    frame_data = req_data['frameData']
    target_directory = req_data['targetDirectory']
    
    # Compare the faces
    matched_target_image = compare_faces(frame_data, target_directory)
    
    # Return the result
    return matched_target_image
 

def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
clf = joblib.load('models/face_spoofing.pkl')


def process_base64_image(base64_image):
    try:
        image = base64.b64decode(base64_image.split(',')[1])
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def detect_face_spoofing(base64_image):
    img = process_base64_image(base64_image)

    if img is None:
        return None

    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    faces3 = net.forward()

    measures = np.zeros(1, dtype=np.float)

    if faces3 is not None:
        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                roi = img[y:y1, x:x1]

                point = (0, 0)

                img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
                img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

                ycrcb_hist = calc_hist(img_ycrcb)
                luv_hist = calc_hist(img_luv)

                feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
                feature_vector = feature_vector.reshape(1, len(feature_vector))

                prediction = clf.predict_proba(feature_vector)
                prob = prediction[0][1]

                measures[0] = prob

                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

                point = (x, y-5)
                
                if 0 not in measures:
                    if np.mean(measures) >= 0.7:
                        return "Spoof"
                    else:
                        return "Real"
    else:
        return None


@app.route('/spoofing_detection_multiple_images', methods=['POST'])
def spoofing_detection_multiple_images():
    req_data = request.get_json(force=True)
    
    images = req_data['images'] # extracting base64 images
    
    count = 0
    for image in images:
        result = detect_face_spoofing(image)

        if result == "Real":
            count = count + 1
    
    if count >= 15:
        return "Real"
    else:
        return "Spoof"


@app.route('/')
def index():
    return "App is running fine"


if __name__ == '__main__':
    app.run()
