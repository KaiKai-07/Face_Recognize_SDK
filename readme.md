# Face_Recognize_SDK
---
## 以下為環境安裝（已經安裝完python）：
需要安裝的模組：
* cv2
* dlib
* numpy
* openvino
```bash
sudo apt update
sudo apt install build-essential cmake  
pip install -r requirements.txt
```

安裝完後把`FD_SDK.py`(或`FD_SDK.so`)移至與主程式同一目錄並在主程式`from FD_SDK import FaceDetection`即可使用api
如要使用人臉記憶功能請先`python3 create_db.py`

---
## API功能說明(以下如有參數是帶入圖片的都請先cv2.imread後再帶入)
### 初始化物件 FaceDetection(engine = '模型')
功能：帶入模型即可啟動模型，目前支援opencv和openvino  
範例：
```python
face_detection = FaceDetection(engine='opencv')
```
### img_detect(img)
功能：從圖片偵測人臉,並回傳一個list,裡面是每張臉的邊界座標[x1,y1,x2,y2]  
範例：
```python
face = face_detection.img_detect(img)
```
### detect_with_cam()
功能：自動啟動攝影機並偵測畫面中的人臉,回傳攝影機畫面與一個list,list裡面是每張臉的邊界座標[x1,y1,x2,y2]  
範例：
```python
frame, face = face_detection.detect_with_cam(img)
```
### recognize_single(ref_img,img)
功能：傳入兩張照片,第一張是註冊者的照片，第二張是欲進行人臉辨識的照片,兩張照片都只能有一張人臉  
回傳兩個值：
* 第一個值用來判斷兩張照片是否都個只有一張人臉，若是的話回傳0，不是的話回傳-1
* 第二個值是一個list,裡面是臉的相似度和邊界座標[similarity,x1,y1,x2,y2],若第一個值是-1則回傳None   

範例：
```python
ret, face = face_detection.recognize_single(ref_img,img)
```
### recognize_multi(ref_img,img)
功能：傳入兩張照片,第一張是註冊者的照片，第二張是欲進行人臉辨識的照片,第一張照片只能有一張人臉  
回傳兩個值：
* 第一個值用來判斷第一張照片是否只有一張人臉，若是的話回傳0，不是的話回傳-1
* 第二個值是一個list,裡面是每張臉的相似度和邊界座標[similarity,x1,y1,x2,y2],若第一個值是-1則回傳None  

範例：
```python
ret, face = face_detection.recognize_single(ref_img,img)
```
### recognize_with_cam(ref_img)
功能：傳入兩張照片,第一張是註冊者的照片，第二張是欲進行人臉辨識的照片,第一張照片只能有一張人臉  
回傳三個值：
* 第一個值用來判斷照片是否只有一張人臉，若是的話回傳0，不是的話回傳-1,且第二第三個直接為None
* 第二個值為攝影機畫面
* 第三個值是一個list,裡面是每張臉的相似度和邊界座標[similarity,x1,y1,x2,y2]

範例：
```python
ret, frame, face = face_detection.recognize_with_cam(ref_img)
```
### add_face(name,img)
功能：呼叫者提供一張照片及名字進行註冊，SDK會把臉部特徵存、名字跟一個具有唯一性的ID存在資料庫中，註冊成功會回傳此ID,否則回傳-1  
範例：
```python
id = face_detection.add_face('Chou',img)
```
### modify_face(id,name = 'newname',img = 'newimg')
功能：呼叫者提供ID，再提供照片或名字即可把資料庫的內容更新(若選照片或名字擇一座更改)  
範例：
```python
face_detection.modify_face(1, name = 'Chou', img = new_img)
face_detection.modify_face(1, ame = 'Chou') #只改名字
face_detection.modify_face(1, img = new_img) #只改照片
```
### delete_face(id)
功能：呼叫者提供ID進行資料庫中資料的刪除  
範例：
```python
face_detection.delete_face(1)
```
### auto_recognize():
功能：自動啟動攝影機,偵測畫面中的人臉並與資料庫中的人臉做比對  
回傳兩個值：
* 第一個值為攝影機畫面
* 第二個值是一個list，裡面是每張臉的名字、最高相似度和邊界座標[name,max_similarity,x1,y1,x2,y2],若最高相似度小於0.5則名字回傳'Unknown'  
範例：
```python
frame, face = face_detection.auto_recognize()
```
