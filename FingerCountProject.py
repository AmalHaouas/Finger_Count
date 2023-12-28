import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Change the index to 0 if you have only one camera
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlaysList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlaysList.append(image)

print(len(overlaysList))
pTime = 0

detector = htm.handDetector(detectionCon=1)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) !=0:
        fingers=[]

        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        #print(totalFingers)

        h,w,c=overlaysList[totalFingers-1].shape
        img[0:h,0:w]= overlaysList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(250,255,205),cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 395), cv2.FONT_HERSHEY_PLAIN, 10, (200, 150, 200), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        break

    if success and img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cv2.imshow("Image", img)
    else:
        print("Error: Invalid image.")

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
