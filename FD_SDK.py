import cv2
import dlib
import numpy as np
from openvino.runtime import Core

class FaceDetection:
    def __init__(self,engine):
        self.engine = engine
        if self.engine == 'opencv':
            self.detector = dlib.get_frontal_face_detector()
        elif self.engine == 'openvino':
            self.core = Core()

            # 指定模型的絕對路徑
            self.model = self.core.read_model(
                model="intel/face-detection-0204/FP32/face-detection-0204.xml",
                weights="intel/face-detection-0204/FP32/face-detection-0204.bin"
            )
            self.compiled_model = self.core.compile_model(self.model, "CPU")

            # 取得模型輸入輸出資訊
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            self.b, self.c, self.h, self.w = self.input_layer.shape  # 模型輸入大小
        self.cap = cv2.VideoCapture(0)

    def detect_with_cam(self):
        if self.engine == 'opencv':
            ret, frame = self.cap.read()
            rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 偵測人臉
            face_rects, scores, idx = self.detector.run(rgbframe, 0,0)

            # 取出所有偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                text = "%2.2f(%d)" % (scores[i], idx[i])

                # 以方框標示偵測的人臉
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

                # 標示分數
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 顯示結果
            cv2.imshow("Face Detection (OpenCV)", frame)

        elif self.engine == 'openvino':
            ret, frame = self.cap.read()

            # 前處理：resize、轉換成 BCHW
            image_resized = cv2.resize(frame, (self.w, self.h))
            image_input = image_resized.transpose((2, 0, 1))  # HWC -> CHW
            image_input = image_input[np.newaxis, :].astype(np.float32)  # 增 batch 維度

            results = self.compiled_model([image_input])[self.output_layer]

            # 繪製預測框
            for detection in results[0][0]:
                conf = float(detection[2])
                if conf > 0.9:
                    x_min = int(detection[3] * frame.shape[1])
                    y_min = int(detection[4] * frame.shape[0])
                    x_max = int(detection[5] * frame.shape[1])
                    y_max = int(detection[6] * frame.shape[0])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{conf:.2f}"
                    cv2.putText(
                        frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # 顯示畫面
            cv2.imshow('Face Detection (OpenVINO)', frame)

    def img_detect(self,image):
        img = cv2.imread(image)
        if self.engine == 'opencv':
            # 偵測人臉
            rgbimg =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            face_rects, scores, idx = self.detector.run(rgbimg, 2, 0)

            # 取出所有偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                text = "%2.2f(%d)" % (scores[i], idx[i])

            # 以方框標示偵測的人臉
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 顯示結果
            cv2.imshow("Face Detection (OpenCV)", img)
        elif self.engine == 'openvino':
            # 前處理：resize、轉換成 BCHW
            image_resized = cv2.resize(img, (self.w, self.h))
            image_input = image_resized.transpose((2, 0, 1))  # HWC -> CHW
            image_input = image_input[np.newaxis, :].astype(np.float32)  # 增 batch 維度

            results = self.compiled_model([image_input])[self.output_layer]

            # 繪製預測框
            for detection in results[0][0]:
                conf = float(detection[2])
                if conf > 0.9:
                    x_min = int(detection[3] * img.shape[1])
                    y_min = int(detection[4] * img.shape[0])
                    x_max = int(detection[5] * img.shape[1])
                    y_max = int(detection[6] * img.shape[0])
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    label = f"{conf:.2f}"
                    cv2.putText(
                        img, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # 顯示畫面
            cv2.imshow('Face Detection (OpenVINO)', img)

    def DetectionEnd(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detection = FaceDetection(engine='opencv')

    while True:
        #face_detection.detect_with_cam()
        face_detection.img_detect('chou.jpg')
        # 檢查是否按下 'q' 鍵退出
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    face_detection.DetectionEnd()
