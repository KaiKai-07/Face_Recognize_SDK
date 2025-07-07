from FD_SDK import FaceDetection
import cv2
import sqlite3

face_detection = FaceDetection(engine='opencv')
'''
#新增人臉
img = cv2.imread("image/chou.jpg")
id = face_detection.add_face("chou",img)
print(id)

conn = sqlite3.connect("face.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
cur.execute("SELECT feature_type, feature FROM face WHERE id = ?",(id,))
f1, f2 = cur.fetchone()
print(id, type(id))

print(f1)
#修改人臉
img = cv2.imread("image/chou.jpg")
face_detection.modify_face(1,name = 'chou', img = img)
conn = sqlite3.connect("face.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
cur.execute("SELECT feature_type, feature FROM face WHERE id = 1")
f1, f2 = cur.fetchone()
print(id, type(id))

print(f1)
print(f2)
'''
while True:
    frame, bbox = face_detection.auto_recognize()
    for box in bbox:
        name, dist, x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
                frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    #face_detection.img_detect('chou.jpg')
    cv2.imshow("Face Detection (OpenCV)", frame)
    # 檢查是否按下 'q' 鍵退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

face_detection.DetectionEnd()

