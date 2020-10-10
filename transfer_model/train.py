import keras
import warnings
import tensorflow as tf
import os
import sys
import multiprocessing
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import InceptionResNetV2, Xception, ResNet50
from keras.models import Model, load_model
from keras import Input
from keras.layers import Dense, Dropout
from PIL import ImageFile
from image_gen_extended import ImageDataGenerator
from datetime import datetime

#当前时间
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


# 定义的文件路径
root_path =  r'G:\13classes_transfer'
train_path = r'G:\13classes_transfer\train'
valid_path = r'G:\13classes_transfer\valid'
model_name = "ResNet50.keras"
model_save_path = root_path + '/' + model_name + now +'/' 
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

batch_size = 20
epochs = 64


# 解决读取截断图片问题，设为True
ImageFile.LOAD_TRUNCATED_IMAGES = True

# keras回调函数，可以使用回调函数来查看训练模型的内在状态和统计等，
#       现在只用了回调函数的on_epoch_end终止训练


class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # on_epoch_end: logs include `acc` and `loss`, and
        #    logs[]包括'acc'和'loss',还有'val_loss'、'val_acc'
        print("----train logs epoch accuracy----",
              logs['acc'], '-------------------------------', sep='\n')
        if logs['acc'] > 0.95:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('------train end------')


def train(train_path, valid_path, model_save_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 给gpu排序
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 选择gpu设备

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # 过滤警告提醒

    config = tf.ConfigProto(device_count={
                            "CPU": 16}, inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)  # 设置CPU最大数量为16
    config.gpu_options.allocator_type = 'BFC'  # 使用BFC算法 BFC是分配、释放内存，碎片管理的算法
    config.gpu_options.allow_growth = True  # 程序按需申请内存

    with tf.Session(config=config):
        pool = multiprocessing.Pool(processes=16)  # 创建16个进程
        img_datagen = ImageDataGenerator(  # 图像预处理
            rescale=1./255,             #对图像进行放大、缩小
            rotation_range=2,           #随机旋转
            width_shift_range=0.2,      #沿着水平、垂直方向为变化范围进行平移
            height_shift_range=0.2,
            zoom_range=[0.9, 1.2],      #按比例随机缩放图像尺寸
            fill_mode='nearest',        #填充像素
            dim_ordering='tf',
            pool=pool
        )
        train_generator = img_datagen.flow_from_directory(          #以文件夹路径为参数，生成经过数据提升/归一化数据，在一个无限循环中产生无限的batch数据
            train_path,
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical'
        )
        valid_generator = img_datagen.flow_from_directory(
            valid_path,
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical'
        )

        early_stopping = EarlyStopping(                     #防止模型过拟合，当网络在训练集表现地越来越好，loss表现地越来越差地时候
            monitor='val_acc', mode='max', min_delta=0.01, patience=8, verbose=1
        )

        auto_lr = ReduceLROnPlateau(                        #当标准评估停止提升时，降低学习速率
            monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', eosilon=0.0001, cooldown=0, min_lr=0
        )

        callback_checkpoint = ModelCheckpoint(filepath=model_save_path + "{epoch:03d}" + model_name,            #在每个训练期之后保存模型
                                              monitor='val_loss',
                                              mode='min',
                                              verbose=1,
                                              save_weights_only=False,
                                              save_best_only=False)

        model_path = ''
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            with tf.device("/cpu:0"):
                base_model = ResNet50(                  #加载ResNet50的Image Net的预训练模型
                    include_top=True, weights='imagenet', input_tensor=None, input_shape=None
                )
                base_model = Model(                     #设置模型的输入与输出
                    inputs=[base_model.input], outputs=[base_model.get_layer('avg_pool').output], name='InceptionResNetV2'
                )

            img = Input(shape=(299, 299, 3), name='img')
            feature = base_model(img)

            classes_count = len(os.listdir(train_path))
            category_predict = Dense(classes_count, activation='softmax', name='ctg_out')(
                Dropout(0.5)(feature)
            )

            model = Model(inputs=img, outputs=category_predict)
            
            #设置冻结层
            for layers in base_model.layers[:20]:
                layers.trainable = False
            for layers in base_model.layers[20:]:
                layers.trainable = True
            
            print('---------base model layers---------','{}\t{}'.format(datetime.now().strftime('%m:%d %H-%M-%S'),len(base_model.layers)), sep='\n')
            print('---------train generator count---------','{}\t{}'.format(datetime.now().strftime('%m:%d %H-%M-%S'),len(train_generator.filenames)), sep='\n')
            
            optimizer = keras.optimizers.Adadelta()
            model.compile(optimizer=optimizer,
                            loss = {
                                'ctg_out':'categorical_crossentropy'
                            },
                            metrics=['accuracy'])
            each_epoch_callback = CustomCallback()

            model.fit_generator(train_generator,
                                steps_per_epoch=len(train_generator.filenames) // (3*batch_size),
                                epochs=epochs,
                                verbose=1,
                                workers=16,
                                validation_data=valid_generator,
                                validation_steps=len(valid_generator.filenames)//(batch_size),
                                callbacks=[callback_checkpoint, early_stopping, auto_lr, each_epoch_callback]
                                )


if __name__ == "__main__":
    train(train_path, valid_path, model_save_path)
