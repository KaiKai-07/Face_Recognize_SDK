from FD_SDK import FaceDetection
import cv2

face_detection = FaceDetection(engine='openvino')

while True:
    #frame, boxes = face_detection.detect_with_cam()
    img1 = cv2.imread('image/kai2.jpg')
    #img2 = cv2.imread('chou2.jpg')
    ret, frame, bbox = face_detection.recognize_with_cam(img1)
    if ret != 0:
        print('error')
        break
    #print(bbox)
    
    for box in bbox:
        dist, x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
                frame, f"({dist:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
    #face_detection.img_detect('chou.jpg')
    cv2.imshow("Face Detection (OpenCV)", frame)
    # 檢查是否按下 'q' 鍵退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

face_detection.DetectionEnd()

