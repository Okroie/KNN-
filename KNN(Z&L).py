# *_* coding: UTF-8 *_*
# Team: BigData Two
# Time: 2022/4/9 21:43
# Name: 偶颗
# Program: zl-KC.PY
# Format: PyCharm

'''                                                 KNN算法                                                           '''
'''
什么是算法 ？
算法就是：解题方案的准确而完整的描述

什么是KNN算法 ？
邻近算法，或者说K最邻近（KNN，K-NearestNeighbor）分类算法是数据挖掘分类技术中最简单的方法之一。
所谓K最近邻，就是K个最近的邻居的意思，说的是每个样本都可以用它最接近的K个邻近值来代表。
近邻算法就是将数据集合中每一个记录进行分类的方法。

KNN算法简介：
KNN（K- Nearest Neighbor）法即K最邻近法，最初由 Cover和Hart于1968年提出，是一个理论上比较成熟的方法，也是最简单的机器学习算法之一。
该方法的思路非常简单直观：如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。
该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。
该方法的不足之处是计算量较大，因为对每一个待分类的文本都要计算它到全体已知样本的距离，才能求得它的K个最邻近点。
目前常用的解决方法是事先对已知样本点进行剪辑，事先去除对分类作用不大的样本。
另外还有一种 Reverse KNN法，它能降低KNN算法的计算复杂度，提高分类的效率 。
KNN算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算法比较容易产生误分。
'''

# 说得简单点，它就是一个跟随大流，没有自己主见的算法。

# # # 用Python实现KNN算法
# import numpy, operator
# def meta():
#     data = numpy.array([[20, 0], [19, 1], [17, 3], [1, 19], [2, 18], [3, 17]])
#     type = ['好分数', '好分数', '好分数', '继续努力', '继续努力', '继续努力']
#     return data, type
# def KNN(a, b, c, k):
#     x = b.shape[0]
#     dt = (numpy.tile(a, (x, 1)) - b) ** 2
#     ad = dt.sum(axis=1)
#     sd = ad ** 0.5
#     ed = sd.argsort()
#     dc = {}
#     for i in range(k):
#         w = c[ed[i]]
#         dc[w] = dc.get(w, 0) + 1
#     ndc = sorted(dc.items(), key=operator.itemgetter(1), reverse=True)
#     return ndc[0][0]
# if __name__ == '__main__':
#     d, t = meta()
#     it = [12, 8]
#     print('输入数据对应的类型是: {}'.format(KNN(it, d, t, 3)))

# conda create -n face python=3.7 # 虚拟环境名为face，使用Python3.7
# activate face # 激活环境


'''                                             人脸识别                                                               '''
# 人脸识别，是基于人的脸部特征信息进行身份识别的一种生物识别技术。
# 用摄像机或摄像头采集含有人脸的图像或视频流，并自动在图像中检测和跟踪人脸，进而对检测到的人脸进行脸部识别的一系列相关技术，通常也叫做人像识别、面部识别。
#
# 介绍：
# 人脸识别系统的研究始于20世纪60年代，80年代后随着计算机技术和光学成像技术的发展得到提高，而真正进入初级的应用阶段则在90年后期，并且以美国、德国和日本的技术实现为主；
# 人脸识别系统成功的关键在于是否拥有尖端的核心算法，并使识别结果具有实用化的识别率和识别速度；
# “人脸识别系统”集成了人工智能、机器识别、机器学习、模型理论、专家系统、视频图像处理等多种专业技术，同时需结合中间值处理的理论与实现，
# 是生物特征识别的最新应用，其核心技术的实现，展现了弱人工智能向强人工智能的转化

# face_recognition, dlib（cmake、Boost）   face_recognition是基于dlib的深度学习人脸识别库
# opencv，  PIL --> Pillow


# #  识别图片中的所有人脸并显示出来
# from PIL import Image
# import face_recognition
# # 将jpg文件加载到numpy 数组中
# image = face_recognition.load_image_file("hd.com.jpg")
# # 使用默认的给予HOG模型查找图像中所有人脸
# # 这个方法已经相当准确了，但还是不如CNN模型那么准确，因为没有使用GPU加速
# # 另请参见: find_faces_in_picture_cnn.py
# face_locations = face_recognition.face_locations(image)
# # 使用CNN模型
# # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
# # 打印：我从图片中找到了 多少 张人脸
# print("I found {} face(s) in this photograph.".format(len(face_locations)))
# # 循环找到的所有人脸
# for face_location in face_locations:
#         # 打印每张脸的位置信息
#         top, right, bottom, left = face_location
#         print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
# # 指定人脸的位置信息，然后显示人脸图片
#         face_image = image[top:bottom, left:right]
#         pil_image = Image.fromarray(face_image)
#         pil_image.show()


# 1 人脸 静态
# r'D:\PyCharm\pythonProject\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'


# import cv2 as cv
# def x():
#     # 将图片转换为灰度图片
#     gray = cv.cvtColor(rs, cv.COLOR_BGR2GRAY)
#     # 加载特征数据
#     fd = cv.CascadeClassifier(r'D:\PyCharm\pythonProject\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
#     faces = fd.detectMultiScale(gray)
#     for x, y, w, h in faces:
#         cv.rectangle(rs, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
#     cv.imshow('photo', rs)
# # 加载图片
# img = cv.imread('hd.jpg')
# # 将图片缩小至原来的1/2
# height, width = img.shape[:2]
# rs = cv.resize(img, (int(width / 3), int(height / 3)), interpolation=cv.INTER_CUBIC)
# x()
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2 as cv
# def x():
#     # 将图片灰度
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # 加载特征数据
#     fd = cv.CascadeClassifier(
#         r'D:\PyCharm\pythonProject\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
#     faces = fd.detectMultiScale(gray)
#     for x, y, w, h in faces:
#         print(x, y, w, h)
#         cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
#         cv.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=2)
#     # 显示图片
#     cv.imshow('result', img)
# # 加载图片
# img = cv.imread('kb.webp')
# # 调用人脸检测方法
# x()
# cv.waitKey(0)
# cv.destroyAllWindows()


# 人脸动态
# import cv2 as cv
# def x(img):
#     # 将图片灰度
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # 加载特征数据
#     fd = cv.CascadeClassifier(
#         r'D:\PyCharm\pythonProject\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
#     faces = fd.detectMultiScale(gray)
#     for x, y, w, h in faces:
#         cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
#         cv.circle(img, center=(x + w // 2, y + h // 2), radius=(w // 2), color=(0, 255, 0), thickness=2)
#     cv.imshow('result', img)
# # 读取视频
# c = cv.VideoCapture('KobeBryant.mp4')
# while True:
#     flag, frame = c.read()
#     if not flag:
#         break
#     x(frame)
#     # 按0结束
#     if ord('0') == cv.waitKey(10):
#         break
# cv.destroyAllWindows()
# c.release()


# import cv2
# im = cv2.imread('hd.jpg', 1)
# fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces = fd.detectMultiScale(im, scaleFactor=1.3, minNeighbors=5)
# for (x, y, w, h) in faces:
#     im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
# cv2.imshow('img', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 人脸静态
# 导入opencv-python


# import cv2
# # 读入一张图片，引号里为图片的路径，需要你自己手动设置
# im = cv2.imread('rc1.png', 1)
# # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
# fd = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
# faces = fd.detectMultiScale(im, scaleFactor=1.3, minNeighbors=5)
# # 对每一张脸，进行如下操作
# for (x, y, w, h) in faces:
#     # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
#     im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
# # 在"photo"窗口中展示效果图
# cv2.imshow('photo', im)
# # 监听键盘上任何按键，如有按键即退出并关闭窗口，并将图片保存为aaa.jpg
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('aaa.png', im)


# # # 人脸实时
# import cv2
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# # 调用摄像头摄像头
# cap = cv2.VideoCapture(0)
# while 1:
#     # 获取摄像头拍摄到的画面
#     ret, frame = cap.read()
#     faces = face_cascade.detectMultiScale(frame, 1.3, 5)
#     img = frame
#     for (x, y, w, h) in faces:
#         # 画出人脸框，蓝色，画笔宽度微
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
#         face_area = img[y: y + h, x: x + w]
#         eyes = eye_cascade.detectMultiScale(face_area)
#         # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
#         for (ex, ey, ew, eh) in eyes:
#             # 画出人眼框，绿色，画笔宽度为1
#             cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
#     # 实时展示效果画面
#     cv2.imshow('frame2', img)
#     # 每5毫秒监听一次键盘动作
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# # 最后，关闭所有窗口q
# cap.release()
# cv2.destroyAllWindows()

# 人脸实时微笑
# import cv2
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# # 调用摄像头摄像头
# cap = cv2.VideoCapture(0)
# while 1:
#     # 获取摄像头拍摄到的画面
#     ret, frame = cap.read()
#     faces = face_cascade.detectMultiScale(frame, 1.3, 2)
#     img = frame
#     for (x, y, w, h) in faces:
#         # 画出人脸框，蓝色，画笔宽度微
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
#         face_area = img[y: y + h, x: x + w]
#         ## 人眼检测
#         # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
#         eyes = eye_cascade.detectMultiScale(face_area, 1.3, 10)
#         for (ex, ey, ew, eh) in eyes:
#             # 画出人眼框，绿色，画笔宽度为1
#             cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
#         ## 微笑检测
#         # 用微笑级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
#         smiles = smile_cascade.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25),
#                                                 flags=cv2.CASCADE_SCALE_IMAGE)
#         for (ex, ey, ew, eh) in smiles:
#             # 画出微笑框，红色（BGR色彩体系），画笔宽度为1
#             cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
#             cv2.putText(img, 'Smile', (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
#     # 实时展示效果画面
#     cv2.imshow('frame2', img)
#     # 每5毫秒监听一次键盘动作
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# # 最后，关闭所有窗口
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy
# import os
# from PIL import Image
#
#
# def getImageAndLabels(path):
#     facesSamples = []
#     ids = []
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # 检测人脸
#     face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     # 遍历列表中的图片
#     for imagePath in imagePaths:
#         # 打开图片
#         PIL_img = Image.open(imagePath).convert('L')
#         # 将图像转换为数组
#         img_numpy = numpy.array(PIL_img, 'uint8')
#         faces = face_detector.detectMultiScale(img_numpy)
#         # 获取每张图片的id
#         id = int(os.path.split(imagePath)[1].split('.')[0])
#         for x, y, w, h in faces:
#             facesSamples.append(img_numpy[y:y + h, x:x + w])
#             ids.append(id)
#     return facesSamples, ids
#
#
# if __name__ == '__main__':
#     # 图片路径
#     path = r'C:\Users\28332\Pictures\f'
#     # 获取图像数组和id标签数组
#     faces, ids = getImageAndLabels(path)
#     # 获取训练对象
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, numpy.array(ids))
#     # 保存文件
#     recognizer.write('train.yml')
