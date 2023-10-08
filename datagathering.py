import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(1)
_, frame = webcam.read()

data = []

while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # cv2.imshow("Hi", frame)


    try:
        x_ratio = round(gaze.horizontal_ratio(), 2)
        y_ratio = round(gaze.vertical_ratio(), 2)
    except TypeError:
        x_ratio = 0
        y_ratio = 0


    # print(f"({x_ratio}, {y_ratio})")

    data.append(f"({round(y_ratio, 2)}, {round(x_ratio, 2)})")

    if len(data) == 60:
        print("dump")

        with open("data.txt", "a") as file:
            file.write(f"{data}\n")

        data = []

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()