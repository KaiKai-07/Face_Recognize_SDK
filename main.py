from FD_SDK import FaceDetection
import cv2
import sqlite3

face_detection = FaceDetection(engine='openvino')

while True:
    print('＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝')
    print('What function would you like to use?')
    print('1.Detect faces from an image')
    print('2.Detect faces using the camera')
    print('3.Recognize a face from an image (only one person in the photo)')
    print('4.Recognize faces from an image (the photo can contain multiple people)')
    print('5.Recognize faces using the camera')
    print('6.Add a face to the database')
    print('7.Modify a face in the database')
    print('8.Delete a face from the database')
    print('9.Recognize faces using those stored in the database')
    print('10.exit')
    choose = int(input())
    if choose == 1:
        img = input('Please provide an image:')
        img = cv2.imread(img)
        boxes = face_detection.img_detect(img)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Face Detection", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choose == 2:
        while True:
            frame, boxes = face_detection.detect_with_cam()
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Face Detection", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    elif choose == 3:
        ref_img = input('Please provide the registered user\'s photo:')
        img = input('Please provide the photo to be recognized:')
        ref_img = cv2.imread(ref_img)
        img = cv2.imread(img)
        ret, boxes = face_detection.recognize_single(ref_img,img)
        if ret != 0 :
            print('more than one face')
            continue
        for box in boxes:
            similarity, x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                    img, f"({similarity:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        cv2.imshow("Face Recognition", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choose == 4:
        ref_img = input('Please provide the registered user\'s photo:')
        img = input('Please provide the photo to be recognized:')
        ref_img = cv2.imread(ref_img)
        img = cv2.imread(img)
        ret, boxes = face_detection.recognize_multi(ref_img,img)
        if ret != 0 :
            print('The registered user\'s photo has more than one face')
            continue
        for box in boxes:
            similarity, x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                    img, f"({similarity:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
        cv2.imshow("Face Recognition", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choose == 5:
        ref_img = input('Please provide the registered user\'s photo:')
        ref_img = cv2.imread(ref_img)
        # if ref_img is None:
        #     print('error')
        while True:
            ret, frame, boxes = face_detection.recognize_with_cam(ref_img)
            if ret != 0 :
                print('The registered user\'s photo has more than one face')
                break
            for box in boxes:
                dist, x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"({dist:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    elif choose == 6:
        name = input('Please provide the registered user\'s name:')
        img = input('Please provide the registered user\'s image:')
        img = cv2.imread(img)
        id = face_detection.add_face(name,img)
        if id == -1:
            print('fail')
        else:
            print('success, id number is',id)

    elif choose == 7:
        id = input('Which id\'s data you want to modify?')
        newname = input('Please provide the new name:')
        new_img = input('Please provide the new image:')
        new_img = cv2.imread(new_img)
        ret = face_detection.modify_face(id = id, name = newname, img = new_img)
        if ret == 0:
            print('modify success')
        else:
            print('fail')
    elif choose == 8:
        id = input('Which id\'s data you want to delete?')
        ret = face_detection.delete_face(id)
        if ret == 0:
            print('delete success')
        else:
            print('fail')
    
    elif choose == 9: 
        while True:
            frame, bbox = face_detection.auto_recognize()
            for box in bbox:
                name, dist, x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                        frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
            cv2.imshow("Face Detection (OpenCV)", frame)
            # 檢查是否按下 'q' 鍵退出
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
    if choose == 10:
        face_detection.DetectionEnd()
        break

