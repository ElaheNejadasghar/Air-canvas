import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

delete_button_x1, delete_button_y1 = 100, 20
delete_button_x2, delete_button_y2 = 200, 80

red_button_x1, red_button_y1 = 250, 20
red_button_x2, red_button_y2 = 350, 80

black_button_x1, black_button_y1 = 400, 20
black_button_x2, black_button_y2 = 500, 80

color = (0,0,0)
points = []

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                if id==4:
                    ccx = cx
                    ccy = cy
                if id==8:

                    if (delete_button_x1 <= cx <= delete_button_x2 and delete_button_y1 <= cy <= delete_button_y2):
                        points.clear()
                    
                    if (red_button_x1 <= cx <= red_button_x2 and red_button_y1 <= cy <= red_button_y2):
                        color = (0,0,255)
                    if (black_button_x1 <= cx <= black_button_x2 and black_button_y1 <= cy <= black_button_y2):
                        color = (0,0,0)
                        
                    distance = math.hypot(ccx - cx, ccy - cy)
                    if distance > 50:
                        points.append((cx,cy,color))
                    else:
                        points.append(None)

                    cv2.circle(frame,(cx,cy),15,color,cv2.FILLED)

                mpDraw.draw_landmarks(
                    frame,
                    handLms,
                    mpHands.HAND_CONNECTIONS
                )

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame,str(int(fps)),(15,45),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

    cv2.rectangle(frame, (delete_button_x1, delete_button_y1),
              (delete_button_x2, delete_button_y2), (255, 0, 255), cv2.FILLED)
    cv2.putText(frame, "DELETE", (105, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.rectangle(frame, (red_button_x1, red_button_y1),
              (red_button_x2, red_button_y2), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "RED", (275, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    cv2.rectangle(frame, (black_button_x1, black_button_y1),
              (black_button_x2, black_button_y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "BLACK", (415, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    for i in range(1,len(points)):
        if points[i-1] is not None and points[i] is not None:
            x1, y1, c1 = points[i-1]
            x2, y2, c2 = points[i]
            cv2.line(frame, (x1, y1), (x2, y2), c2, 3)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()