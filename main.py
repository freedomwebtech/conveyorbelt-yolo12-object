import cv2
from ultralytics import YOLO
import cvzone


model = YOLO('best.pt')
names = model.names

cap = cv2.VideoCapture("pac.mp4")
frame_count = 0




# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True)


    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name=names[class_id]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cvzone.putTextRect(frame,f'{name}',(x1,y1),1,1)
                       
            
     # Show counts
#    cvzone.putTextRect(frame, f'IN: {in_count}', (50, 30), 1, 2, colorR=(0, 255, 0))
#    cvzone.putTextRect(frame, f'OUT: {out_count}', (50, 70), 1, 2, colorR=(0, 0, 255))
#    cv2.line(frame,(line_p1),(line_p2),(0,0,255),2)
    cv2.imshow("RGB", frame)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC to exit
        break
    

cap.release()
cv2.destroyAllWindows()
