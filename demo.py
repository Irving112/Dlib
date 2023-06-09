import os
import cv2
import dlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

RegisterDic = {}
RecognizeDic = {}
cnn_face_detector = dlib.cnn_face_detection_model_v1(r'F:\face_automation\model\mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'F:\face_automation\model\shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1(r"F:\face_automation\model\dlib_face_recognition_resnet_model_v1.dat")

#使用dlib自带的库检测时 带有角度的人脸和光线比较差的检测不到 所以使用cnn算法来检测（基本都能检测到） 缺点是运行速度特别慢（必须要用pytorch和cuda的gpu运行才能提高速度，但是公司电脑没有独显）
#FD检测
def FD(path):
    images = Image.open(path)
    openImage = cv2.imread(path)
    image = cv2.cvtColor(openImage, cv2.COLOR_BGR2RGB)
    #cnn检测
    # faces = cnn_face_detector(image,0)
    #dlib库检测
    detector = dlib.get_frontal_face_detector()
    faces = detector(image,0)
    return image,faces,images,openImage
#绘制人脸框
def plot_rectangle(image, face,path,face_name):
    name = path.split('\\')[-1]
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), -1)
    namedir = r'F:\face_automation\material\result2\%s'%(face_name)
    if os.path.exists(namedir) == False:
        os.mkdir(namedir)
    cv2.imwrite(r'F:\face_automation\material\result2\%s\%s'%(face_name,name), image)

#绘制白色框
def White_plot():
    for facename,facedata in RecognizeDic.items():
        if facedata != []:
            for i in facedata:
                rect = i[1]
                path = i[0]
                res = FD(path)
                openImage = res[3]
                faces = res[1]
                for face in faces:
                    # face = face.rect
                    if face != rect:
                        plot_rectangle(openImage,face,path,facename)
                    else:
                        name = path.split('\\')[-1]
                        cv2.rectangle(openImage, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 4)
                        namedir = r'F:\face_automation\material\result2\%s' % (facename)
                        path = r'F:\face_automation\material\result2\%s\%s' % (facename, name)
                        if os.path.exists(namedir) == False:
                            os.mkdir(namedir)
                        cv2.imwrite(r'F:\face_automation\material\result2\%s\%s' % (facename, name), openImage)
                        writejson(path,face)
#给检测出的人脸绘制矩形框
def draw(RecognizePath):
    for dir, file, filename in os.walk(RecognizePath):
        for filename in filename:
            if filename.endswith('.jpg'):
                path = dir + os.sep + filename
                faces = FD(path)[1]
                openImage = FD(path)[3]
                plot_rectangle(openImage.copy(),faces,path)
                # plt.figure(figsize=(9, 6))
                # plt.suptitle("face detection with dlib", fontsize=14, fontweight="bold")
                #显示最终的检测效果
                # show_image(img_result, "face detection")
                # plt.show()
#显示图片
def show_image(image, title):
    img_RGB = image[:, :, ::-1]  # BGR to RGB
    plt.title(title)
    plt.imshow(img_RGB)
    plt.axis("off")
# 计算注册照特征值
def get_register_features(RegisterPath):
    for dir, file, filename in os.walk(RegisterPath):
        for filename in filename:
            if filename.endswith('.jpg'):
                path = dir + os.sep + filename
                RegisterName = dir.split('\\')[-1]
                res = FD(path)
                faces = res[1]
                image = res[0]
                feature_list_of_person_x = []
                if len(faces) != 0:
                    for i in faces:
                        # i = i.rect
                        shape = predictor(image, i)
                        face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
                        feature_list_of_person_x.append(face_descriptor)
                # features_mean_person_x = np.zeros(128, dtype=object, order='C')
                if feature_list_of_person_x:
                    features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)
                    RegisterDic[RegisterName] = features_mean_person_x
                    # print(RegisterDic)
                # yield (features_mean_person_x, RegisterName)

#计算欧式距离
def get_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    return np.sqrt(np.sum(np.square(feature_1 - feature_2)))

#注册照识别照比对
def compare_face_fatures_with_database(RecognizePath):
    for face_name, face_data in RegisterDic.items():
        RecognizeDic[face_name] = []
        for dir, file, filename in os.walk(RecognizePath):
            for filename in filename:
                if filename.endswith('.jpg'):
                    path = dir + os.sep + filename
                    res = FD(path)
                    faces = res[1]
                    image = res[0]
                    openimage = res[3]
                    if len(faces) != 0:
                            for i in range(len(faces)):
                                face = faces[i]
                                # face = face.rect
                                shape = predictor(image, face)
                                face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
                                    #比对人脸特征，当距离小于 0.4 时认为匹配成功
                                dist = get_euclidean_distance(face_descriptor, face_data)
                                dist = round(dist, 4)
                                if dist < 0.4:
                                    plot_rectangle(openimage.copy(),face,path,face_name)
                                    RecognizeDic[face_name].append((path, face))

#人脸坐标输出json
def writejson(path,face):
    jsonpath = path.replace('.jpg','.json')
    with open(jsonpath,'w',encoding='utf-8') as f:
        f.writelines(str(face))
        print(path)

