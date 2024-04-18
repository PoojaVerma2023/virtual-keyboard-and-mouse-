import cv2
import mediapipe as mp
import time
import math
#PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode

        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                   self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw= True ):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0],bbox[1]),
                              (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox

    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length,img, [x1, y1, x2, y2, cx, cy]

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":

    import autopy
    import pyautogui
    import mediapipe

    import numpy as np
    import pyHandTrackingModule as htm
    from pynput.keyboard import Key, Controller

    from time import sleep
    import mediapipe as mp

    wCam, hCam = 640, 480
    frameR = 100
    smoothening = 7

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    print(wScr, hScr)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[16][1:]
            x2, y2 = lmList[12][1:]

            # print(x1, y1, x2, y2)

            fingers = detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (0, 0, 0), 2)
            # 4. Only index Finger :Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. convert coordinates

                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8.  Index  fold are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 5, img)
                print(length)
                if length < 90:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
                # 9. Thumb finger fold : Right click
                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(4, 2, img)
                print(length)
                if length < 60:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    pyautogui.rightClick()

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(20, 17, img)
                print(length)
                if length < 39:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    pyautogui.doubleClick()

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(16, 13, img)
                print(length)
                if length < 10:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    exit()
            # keyboard operations using index and thumb fingure

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 4, img)
                print(length)
                if length < 60:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)

                    arr = []
                    keyLayout = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
                                 ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
                                 ["a", "s", "d", "f", "g", "h", "j", "k", "l", "."],
                                 ["z", "x", "c", "v", "b", "n", "m", "?", "+", ":"],
                                 [" ", "<", "=", "-", "C", "E", "(", ")", "_", "{"],
                                 ["}", "[", "]", "!", ">", "#", "$", "%", "'", "^"],
                                 ["&", "*", "@"]]



                    mpHands = mp.solutions.hands
                    mpDraw = mp.solutions.drawing_utils
                    Hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
                    keyboard = Controller()
                    mpDraw = mp.solutions.drawing_utils




                    class settings():
                        def __init__(self, pos, size, text):
                            self.pos = pos
                            self.size = size
                            self.text = text




                    def keyboardEdit(img, storedVar):
                        for button in storedVar:
                            x, y = button.pos
                            w, h = button.size

                            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 0, 0), 5)
                            cv2.putText(img, button.text, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

                        return img


                    for i in range(len(keyLayout)):
                        for j, key in enumerate(keyLayout[i]):
                            arr.append(settings([60 * j + 10, 60 * i + 10], [50, 50], key))

                    while True:
                        success, img = cap.read()
                        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = Hands.process(imgRGB)

                        printer = []

                        if results.multi_hand_landmarks:
                            for hand_in_frame in results.multi_hand_landmarks:
                                mpDraw.draw_landmarks(img, hand_in_frame, mpHands.HAND_CONNECTIONS)

                            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                printer.append([cx, cy])

                            if printer:
                                for button in arr:
                                    x, y = button.pos
                                    w, h = button.size

                                    if x < printer[8][0] < x + w and y < printer[8][1] < y + h:
                                        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255),
                                                      cv2.FILLED)
                                        x1, y1 = printer[8][0], printer[8][1]
                                        x2, y2 = printer[12][0], printer[12][1]
                                        clicked = math.hypot(x2 - x1 - 20, y2 - y1 - 20)
                                        print()
                                        if button.text == "<":
                                            keyboard.press(Key.backspace)
                                            keyboard.release(Key.backspace)
                                        if button.text == "C":
                                            keyboard.press(Key.caps_lock)
                                            keyboard.release(Key.caps_lock)
                                        if button.text == "E":
                                            keyboard.press(Key.enter)
                                            keyboard.release(Key.enter)
                                        if button.text == ">":
                                            keyboard.press(Key.space)
                                            keyboard.release(Key.space)

                                        if not clicked > 50:
                                            keyboard.press(button.text)
                                            cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0),
                                                          cv2.FILLED)
                                            sleep(0.15)

                            img = keyboardEdit(img, arr)

                        cv2.imshow("Keyboard", img)
                        if cv2.waitKey(1) & 0xff == ord(" "):
                            break

            # volume control operation
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(20, 4, img)
                print(length)
                if length < 35:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)

                    mp_drawing = mp.solutions.drawing_utils
                    mp_hands = mp.solutions.hands
                    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

                    # Constants for volume control
                    VOLUME_RANGE = 100
                    VOLUME_MIN = 0
                    VOLUME_MAX = VOLUME_RANGE

                    # OpenCV window settings
                    WINDOW_NAME = 'Volume Control'
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

                    cap = cv2.VideoCapture(0)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(frame_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                for idx, landmark in enumerate(hand_landmarks.landmark):
                                    if idx == 4:  # Thumb tip
                                        thumb_tip_x, thumb_tip_y = int(landmark.x * frame.shape[1]), int(
                                            landmark.y * frame.shape[0])
                                    if idx == 8:  # Index finger tip
                                        index_tip_x, index_tip_y = int(landmark.x * frame.shape[1]), int(
                                            landmark.y * frame.shape[0])

                                # Calculate distance between thumb tip and index finger tip
                                distance = math.sqrt(
                                    (thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)

                                # Map distance to volume range
                                volume = int((distance / frame.shape[1]) * VOLUME_RANGE)

                                # Set system volume using pyautogui
                                pyautogui.press('volumedown', presses=volume)
                                pyautogui.press('volumeup', presses=volume)

                        cv2.imshow(WINDOW_NAME, frame)
                        cv2.waitKey(1)
                        break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Mouse", img)
        if cv2.waitKey(1) & 0xff == ord('-'):
            break
