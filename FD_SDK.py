import cv2
import dlib
import numpy as np
from openvino.runtime import Core

class FaceDetection:
    def __init__(self,engine):
        self.engine = engine
        if self.engine == 'opencv':
            self.detector = dlib.get_frontal_face_detector()
            self.sp = dlib.shape_predictor("dlib_model/shape_predictor_68_face_landmarks.dat")  #讀入人臉特徵點模型
            self.facerec = dlib.face_recognition_model_v1("dlib_model/dlib_face_recognition_resnet_model_v1.dat")  #讀入人臉辨識模型
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

            self.landmark_model = self.core.read_model(
                model="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
                weights="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin"
            )
            self.landmark_compiled_model = self.core.compile_model(self.landmark_model, "CPU")

            self.land_input = self.landmark_compiled_model.input(0)
            self.land_output = self.landmark_compiled_model.output(0)
            b_l, c_l, self.h_l, self.w_l = self.land_input.shape

            self.rec_model = self.core.read_model(
                model="intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml",
                weights="intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin"
            )
            self.rec_compiled_model = self.core.compile_model(self.rec_model, "CPU")

            self.rec_input = self.rec_compiled_model.input(0)
            self.rec_output = self.rec_compiled_model.output(0)
            b_r, c_r, self.h_r, self.w_r = self.rec_input.shape  
        self.cap = cv2.VideoCapture(0)

    def get_feature_opencv(self,img,det):
        shape = self.sp(img, det)  #特徵點偵測
        feature = self.facerec.compute_face_descriptor(img, shape)  #取得128維特徵向量
        return np.array(feature)  #轉換numpy array格式
    
    def get_feature_openvino(self,img):
        # 提取參考人臉的 landmarks 並校正
        ref_face_crop = cv2.resize(img, (self.w_l, self.h_l))
        ref_face_input = ref_face_crop.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
        ref_landmarks = self.landmark_compiled_model([ref_face_input])[self.land_output].reshape(-1, 2)
        ref_landmark = ref_landmarks * np.float32([self.w_l, self.h_l])

        landmark_reference = np.float32([
            [0.31556875000000000, 0.4615741071428571],
            [0.68262291666666670, 0.4615741071428571],
            [0.50026249999999990, 0.6405053571428571],
            [0.34947187500000004, 0.8246919642857142],
            [0.65343645833333330, 0.8246919642857142]
        ])
        landmark_ref = landmark_reference * np.float32([self.w_l, self.h_l])
        M = cv2.getAffineTransform(ref_landmark[0:3], landmark_ref[0:3])
        ref_align = cv2.warpAffine(ref_face_crop, M, (self.w_l, self.h_l))

        # 提取參考人臉特徵
        ref_face_rec = cv2.resize(ref_align, (self.w_r, self.h_r))
        ref_face_rec_input = ref_face_rec.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
        ref_embedding = self.rec_compiled_model([ref_face_rec_input])[self.rec_output].flatten()
        return ref_embedding
    
    def detect_with_cam(self):
        if self.engine == 'opencv':
            ret, frame = self.cap.read()
            rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 偵測人臉
            face_rects, scores, idx = self.detector.run(rgbframe, 0,0)
            facevalue = []
            # 取出所有偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                facevalue.append([x1,y1,x2,y2])
    
            return frame ,facevalue
        
        elif self.engine == 'openvino':
            ret, frame = self.cap.read()

            # 前處理：resize、轉換成 BCHW
            image_resized = cv2.resize(frame, (self.w, self.h))
            image_input = image_resized.transpose((2, 0, 1))  # HWC -> CHW
            image_input = image_input[np.newaxis, :].astype(np.float32)  # 增 batch 維度

            results = self.compiled_model([image_input])[self.output_layer]
            facevalue = []
            # 繪製預測框
            for detection in results[0][0]:
                conf = float(detection[2])
                if conf > 0.9:
                    x_min = int(detection[3] * frame.shape[1])
                    y_min = int(detection[4] * frame.shape[0])
                    x_max = int(detection[5] * frame.shape[1])
                    y_max = int(detection[6] * frame.shape[0])
                    facevalue.append([x_min,y_min,x_max,y_max])
            return frame, facevalue

    def img_detect(self,img):
        #img = cv2.imread(image)
        if self.engine == 'opencv':
            # 偵測人臉
            rgbimg =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            face_rects, scores, idx = self.detector.run(rgbimg, 2, 0)
            facevalue = []
            # 取出所有偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                facevalue.append([x1,y1,x2,y2])
    
            return facevalue
        
        elif self.engine == 'openvino':
            # 前處理：resize、轉換成 BCHW
            image_resized = cv2.resize(img, (self.w, self.h))
            image_input = image_resized.transpose((2, 0, 1))  # HWC -> CHW
            image_input = image_input[np.newaxis, :].astype(np.float32)  # 增 batch 維度

            results = self.compiled_model([image_input])[self.output_layer]
            facevalue = []
            # 繪製預測框
            for detection in results[0][0]:
                conf = float(detection[2])
                if conf > 0.9:
                    x_min = int(detection[3] * img.shape[1])
                    y_min = int(detection[4] * img.shape[0])
                    x_max = int(detection[5] * img.shape[1])
                    y_max = int(detection[6] * img.shape[0])
                    facevalue.append([x_min,y_min,x_max,y_max])
            return facevalue

    def recognize_single(self,ref_img,img):
        if self.engine == 'opencv':
            ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            dets = self.detector(ref_img,1)
            if len(dets) != 1 :
                return -1 , None
            dets1 = self.detector(img,1)
            if len(dets1) != 1 :
                return -1 , None
            for det in dets:
                ref_feature = self.get_feature_opencv(ref_img,det)

            for d in dets1:
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                feature = self.get_feature_opencv(img,d)
            dist = np.linalg.norm(ref_feature-feature)

            #facevalue = self.img_detect(img)
            return 0, [[dist,x1,y1,x2,y2]]
        
        elif self.engine == 'openvino':
            ref_facevalue = self.img_detect(ref_img)
            facevalue = self.img_detect(img)
            if len(facevalue) != 1 or len(ref_facevalue) != 1:
                return -1, None
            for result in ref_facevalue:
                x1, y1, x2, y2 = result
                ref_face = ref_img[y1:y2, x1:x2]
                ref_feature = self.get_feature_openvino(ref_face)
            
            for result in facevalue:
                x1, y1, x2, y2 = result
                face = img[y1:y2, x1:x2]
                feature = self.get_feature_openvino(face)
            
            cosine_similarity = np.dot(ref_feature, feature) / (np.linalg.norm(ref_feature) * np.linalg.norm(feature))
            return 0, [[cosine_similarity,x1,y1,x2,y2]]

    def recognize_multi(self,ref_img,img):
        if self.engine == 'opencv':
            ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            box = []
            dets = self.detector(ref_img,1)
            if len(dets) != 1:
                return -1 , None
            for det in dets:
                ref_feature = self.get_feature_opencv(ref_img,det)

            dets = self.detector(img,1)
            for d in dets:
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                feature = self.get_feature_opencv(img,d)
                dist = np.linalg.norm(ref_feature-feature)
                box.append([dist,x1,y1,x2,y2])
            return 0, box
        
        elif self.engine == 'openvino':
            box = []
            ref_facevalue = self.img_detect(ref_img)
            if len(ref_facevalue) != 1:
                return -1, None
            for result in ref_facevalue:
                x1, y1, x2, y2 = result
                ref_face = ref_img[y1:y2, x1:x2]
                ref_feature = self.get_feature_openvino(ref_face)
            
            facevalue = self.img_detect(img)
            for result in facevalue:
                x1, y1, x2, y2 = result
                face = img[y1:y2, x1:x2]
                feature = self.get_feature_openvino(face)
                cosine_similarity = np.dot(ref_feature, feature) / (np.linalg.norm(ref_feature) * np.linalg.norm(feature))
                box.append([cosine_similarity,x1,y1,x2,y2])
            return 0, box
        
    def recognize_with_cam(self,ref_img):
        if self.engine == 'opencv':
            box = []
            ref_img = cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
            dets = self.detector(ref_img,1)
            if len(dets) != 1 :
                return -1, None, None
            for det in dets:
                ref_feature = self.get_feature_opencv(ref_img,det)
            ret, frame = self.cap.read()
            rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets2 = self.detector(rgbframe,1)
            for d in dets2:
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                feature = self.get_feature_opencv(rgbframe,d)
                dist = np.linalg.norm(ref_feature-feature)
                box.append([dist,x1,y1,x2,y2])
            return 0, frame , box
        
        elif self.engine == 'openvino' :
            box = []
            ref_facevalue = self.img_detect(ref_img)
            if len(ref_facevalue) != 1:
                return -1, None, None
            for result in ref_facevalue:
                x1, y1, x2, y2 = result
                ref_face = ref_img[y1:y2, x1:x2]
                ref_feature = self.get_feature_openvino(ref_face)

            ret, frame = self.cap.read()
            facevalue = self.img_detect(frame)
            for result in facevalue:
                x1, y1, x2, y2 = result
                face = frame[y1:y2, x1:x2]
                feature = self.get_feature_openvino(face)
                cosine_similarity = np.dot(ref_feature, feature) / (np.linalg.norm(ref_feature) * np.linalg.norm(feature))
                box.append([cosine_similarity,x1,y1,x2,y2])
            return 0, frame, box
            
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
