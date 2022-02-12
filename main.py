import time
import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 650)
    cap.set(4, 350)
    mode = True
    timing = time.time()

    while True:
        if time.time() - timing > 10.0:
            timing = time.time()  # время с начала эпохи
            mode = not mode

        _, frame1 = cap.read()
        _, frame2 = cap.read()

        colored = red_green(frame1, frame2)
        if mode:
            cv2.putText(colored, 'mode_on', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("camera", colored)
        else:
            cv2.putText(frame1, 'mode_off', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("camera", frame1)

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def red_green(img1, img2):
    """

    Parameters
    ----------
    img1 - The first frame from the video sequence
    img2 - The next frame from the video sequence

    Returns
    -------
    Video sequence with motion recognition

    """
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)  # количество итераций
    nulls = np.zeros((len(img1), len(img1[0])), np.uint8)
    green = cv2.merge([nulls, ~thresh, nulls])
    red = cv2.merge([nulls, nulls, dilated])
    colored = cv2.addWeighted(img1, 1, green, 0.15, 0)
    return cv2.addWeighted(colored, 1, red, 0.5, 0)


if __name__ == "__main__":
    main()
