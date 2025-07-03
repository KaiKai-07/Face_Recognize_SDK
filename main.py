from FD_SDK import FaceDetection
import cv2

face_detection = FaceDetection(engine='opencv')

while True:
    face_detection.detect_with_cam()
    #face_detection.img_detect('chou.jpg')
    # 檢查是否按下 'q' 鍵退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

face_detection.DetectionEnd()

