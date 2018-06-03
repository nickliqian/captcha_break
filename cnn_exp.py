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
from keras.utils.vis_utils import plot_model
from IPython.display import Image
# 计算模型总体准确率
from tqdm import tqdm


# 定义数据生成器
def gen(height, width, n_class, n_len=4, characters="", batch_size=32):
    # X 中有batch_size个三维数组，可以理解为height*width*3的数组
    # 即对应与height*width的彩色图片，每个点由三种颜色像素组成
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    # n_len个n_class*batch_size 4个26*1
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            # 随机字符串，长度为4
            random_str = ''.join([random.choice(characters) for j in range(4)])
            # print(random_str)
            # 每张图片的像素填充X的每个三维数组中，也就是对应于图片像素值的三维数组
            X[i] = generator.generate_image(random_str)
            # y转换为标签-目标值坐标表示法
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        # y [[[],[]...], [[],[]...]]
        #       img   1   2
        # y string1 [[18],[31]]
        #   string2 [[16],[26]]
        #   string3 [[18],[11]]
        #   string4 [[23],[32]]
        yield X, y


# 测试生成器
def decode(y, characters):
    # 获得坐标位的数字,放入一个列表中 y=[11,23,2,5]
    y = np.argmax(np.array(y), axis=2)[:,0]
    # 返回一个字符串
    return ''.join([characters[x] for x in y])


# 计算模型总体准确率
def evaluate(model, height, width, n_class, n_len, characters, batch_num=20):
    batch_acc = 0
    generator = gen(height, width, n_class, n_len, characters)
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        before = decode(y, characters)
        after = decode(y_pred, characters)
        if str(before) == str(after):
            batch_acc += 1
        # batch_acc += np.mean(map(np.array_equal, np.argmax(y, axis=2).T, np.argmax(y_pred, axis=2).T))
    return batch_acc / batch_num


def main():
    # 生成大写字母和数字的组合字符串 >>> '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    characters = string.digits + string.ascii_uppercase

    # 设置图片的参数
    width, height, n_len, n_class = 170, 80, 4, len(characters)

    # 测试生成器
    # 生成的X特征值和y目标值
    X, y = next(gen(height, width, n_class, n_len, characters, batch_size=1))
    # X[i] 是一张图片的像素，直接使用像素生成图片
    plt.imshow(X[0])
    # 添加title
    plt.title(decode(y, characters))

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
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # 网络结构可视化
    plot_model(model, to_file="model.png", show_shapes=True)
    Image('model.png')


    # 训练模型
    # samples_per_epoch 训练集的长度
    # nb_val_samples 验证集的长度
    model.fit_generator(gen(height, width, n_class, n_len, characters),
                        samples_per_epoch=20,
                        nb_epoch=2,
                        validation_data=gen(height, width, n_class, n_len, characters),
                        nb_val_samples=10)

    print("test")
    # 测试模型
    X, y = next(gen(height, width, n_class, n_len, characters))
    y_pred = model.predict(X)
    plt.title('real: %s\npred:%s'%(decode(y, characters), decode(y_pred, characters)))
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')

    # 评估模型准确率
    evaluate(model, height, width, n_class, n_len, characters)

    # 保存模型
    model.save('cnn123.h5')
    # 保存神经网络的结构与训练好的参数
    json_string = model.to_json()  # 等价于 json_string = model.get_config()
    with open('my_model_architecture.json', 'w') as f:
        f.write(json_string)
    model.save_weights('my_model_weights.h5')


if __name__ == '__main__':
    main()