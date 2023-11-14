import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt



def predict(model_path, img_path):
    # 加载已训练好的模型
    model = load_model(model_path)

    # 加载测试图像（请替换为你自己的图像路径）
    img_path = img_path
    img = image.load_img(img_path, target_size=(180, 180))

    # 将图像转换为numpy数组
    img_array = image.img_to_array(img)

    # 添加一个维度，因为模型期望输入是一个批次
    img_array = np.expand_dims(img_array, axis=0)

    # 预处理图像
    # img_array = preprocess_input(img_array)

    # 进行预测
    predictions = model.predict(img_array)
    print(predictions)

    # 获取预测结果
    label_map = ['dog', 'cat']
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # 打印预测结果
    print("Predicted class:", label_map[predicted_class])
    print("Confidence:", confidence)

    # 使用OpenCV将图像加载为BGR格式
    img_bgr = cv2.imread(img_path)
    # 在图像上绘制预测结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, f"l: {label_map[predicted_class]}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, f"p: {confidence}", (10, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # 显示图像
    cv2.imshow("Prediction", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict('model.h5', 'testimage/dog.jpg')
