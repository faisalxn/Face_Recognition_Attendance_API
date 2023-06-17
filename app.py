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
    

@app.route('/')
def index():
    return "App is running fine"


if __name__ == '__main__':
    app.run()
