from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import ast
import random
import requests
import threading
from gaze_tracking import GazeTracking


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Global variables to store the outputs
output = None
distracted_count = 0
looking_count = 0
count = 15

random_data = [
    {"name":"Joshua W",  "status": "", "overall": [12, 3]},
    {"name":"Tristan J", "status": "", "overall": [13, 2]},
    {"name":"Issac I",   "status": "", "overall": [10, 5]},
    {"name":"Mikey G",   "status": "", "overall": [3, 12]},
    {"name":"Javier W",  "status": "", "overall": [9, 6]},
]


def classify(data: list):
    global output, distracted_count, looking_count, count

    data = f"{[ast.literal_eval(cord) for cord in data]}"

    key = "dde15910-63ce-11ee-8144-8724cf42ce1cc663e107-5619-42cd-92c5-28bde1cd03f9"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"

    response = requests.get(url, params={ "data" : data })

    if response.ok:
        responseData = response.json()
        topMatch = responseData[0]
        print(f"{topMatch['class_name']} with {topMatch['confidence']}%")

        if topMatch['class_name'] == 'Distracted':
            distracted_count += 1
            output = False
        elif topMatch['class_name'] == 'Looking':
            looking_count += 1
            output = True

        count += 1

        random_data[0]["status"] = random.choice(["Looking", "Looking", "Looking", "Looking", "Distracted"])
        random_data[0]["overall"][random.choice([0, 0, 0, 0, 1])] += 1
        random_data[0]["lookingpercent"] = round(random_data[0]["overall"][0] / count * 100, 0)

        random_data[1]["status"] = random.choice(["Looking", "Looking", "Looking", "Distracted"])
        random_data[1]["overall"][random.choice([0, 0, 0, 1])] += 1
        random_data[1]["lookingpercent"] = round(random_data[1]["overall"][0] / count * 100, 0)

        random_data[2]["status"] = random.choice(["Looking", "Looking", "Looking", "Distracted", "Distracted"])
        random_data[2]["overall"][random.choice([0, 0, 0, 1, 1])] += 1
        random_data[2]["lookingpercent"] = round(random_data[2]["overall"][0] / count * 100, 0)

        random_data[3]["status"] = random.choice(["Looking", "Distracted", "Distracted", "Distracted", "Distracted"])
        random_data[3]["overall"][random.choice([0, 1, 1, 1])] += 1
        random_data[3]["lookingpercent"] = round(random_data[3]["overall"][0] / count * 100, 0)

        random_data[4]["status"] = random.choice(["Looking", "Looking", "Looking", "Distracted", "Distracted"])
        random_data[4]["overall"][random.choice([0, 0, 0, 1, 1])] += 1
        random_data[4]["lookingpercent"] = round(random_data[4]["overall"][0] / count * 100, 0)

    else:
        response.raise_for_status()

def webcam_processing():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(1)

    data = []
    error = 0

    while True:
        _, frame = webcam.read()

        if frame is not None:
            frame = cv2.flip(frame, 1)

            gaze.refresh(frame)

            try:
                x_ratio = round(gaze.horizontal_ratio(), 2)
                y_ratio = round(gaze.vertical_ratio(), 2)
            except TypeError:
                x_ratio = 0
                y_ratio = 0


            data.append(f"({round(y_ratio, 2)}, {round(x_ratio, 2)})")

            if len(data) == 60:
                thread = threading.Thread(target=classify, args=[data])
                thread.start()

                data = []
        else:
            print("cv2 is dying")
            error += 1

            if error >= 10:
                print("Something went wrong. Do you have a external webcam attached?")
                print("Switching to built in webcam...")
                webcam = cv2.VideoCapture(0)
                print("Switched to built in webcam")

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


# Start the webcam processing in a separate thread
threading.Thread(target=webcam_processing).start()


@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'output': output, 'distracted_count': distracted_count, 'looking_count': looking_count, 'class_list': random_data})


if __name__ == '__main__':
    app.run(port=5000)
