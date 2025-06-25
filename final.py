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

def counter(x,y,x1,y1,x2,y2):
    return(x-x1)*(y2-y1)-(y-y1)*(x2-x1)

line_p1=(432,417)
line_p2=(537,460)

hist={}
counted=set()
in_count=0
out_count=0
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
            if track_id in hist:
               prev_cx,prev_cy=hist[track_id]
               side_1=counter(prev_cx,prev_cy,*line_p1,*line_p2)
               side_2=counter(cx,cy,*line_p1,*line_p2)
               if side_1*side_2 <0 and track_id not in counted:
                   if side_2<0:
                       direction="IN"
                       in_count+=1
                       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                   else:
                       direction="OUT"
                       out_count+=1
                   counted.add(track_id)    
            
            hist[track_id]=(cx,cy)
                       
            
     # Show counts
    cvzone.putTextRect(frame, f'IN: {in_count}', (50, 30), 1, 2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (50, 70), 1, 2, colorR=(0, 0, 255))
    cv2.line(frame,(line_p1),(line_p2),(0,0,255),2)
    cv2.imshow("RGB", frame)
    print(hist)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    

cap.release()
cv2.destroyAllWindows()
