import h5py
from keras.models import model_from_json
import cv2
import numpy as np
import string
from captcha.image import ImageCaptcha
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
# 定义数据生成器导入
from keras.utils.np_utils import to_categorical
# 定义网络结构
from keras.models import *
from keras.layers import *
# 结构可视化
from keras.utils.visualize_util import plot
from IPython.display import Image
# 计算模型总体准确率
from tqdm import tqdm



# 测试生成器
def decode(y, characters):
    # 获得坐标位的数字,放入一个列表中 y=[11,23,2,5]
    y = np.argmax(np.array(y), axis=2)[:,0]
    # 返回一个字符串
    return ''.join([characters[x] for x in y])


def main():
    characters = string.digits + string.ascii_uppercase

    width, height, n_len, n_class = 170, 80, 4, len(characters)

    generator = ImageCaptcha(width=width, height=height)
    img = generator.generate_image("AAAA")
    img.save("ABCD.png")


    #读取model
    # 定义网络结构
    # 特征值结构
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(4):
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)
    # 编译模型
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model = model.load_weights('model.h5')
    image = cv2.imread("./ABCD.png")

    X = np.zeros((1, height, width, 3), dtype=np.uint8)
    X[0] = image

    y_pred = model.predict(X)
    result = decode(y_pred, characters)
    print(result)


if __name__ == '__main__':
    main()