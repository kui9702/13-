from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
import shutil
import datetime
from PIL import ImageFile



# 解决读取截断图片问题，设为True
ImageFile.LOAD_TRUNCATED_IMAGES = True


#当前时间
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

normal_size = 299

#日志文件路径
log_path =  r'G:\13classes_transfer/log'
#模型存储路径
model_path =  r'G:\13classes_transfer/ResNet50.keras/015ResNet50.keras'
#测试文件夹路径
test_path =r'G:\13classes_transfer/test'


if not os.path.exists(log_path):
    os.makedirs(log_path)
    
log_path = os.path.join(log_path,now+'log.txt')


#读取分类ID
def classes_id():
        with open('./train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines


def predict(model_path,path):
    model = load_model(model_path, compile=False) 

    i,total = 0,0
    data_list = []
    for root,dirs,files in os.walk(path):
        for d in dirs:
            every_path = os.path.join(root,d)
            data_list = list(               #将所有图片的信息与路径绑定
                map(lambda x: cv2.resize(cv2.imread(os.path.join(every_path,x)),
                (normal_size,normal_size)),os.listdir(every_path)))
    
            for index, img in enumerate(data_list):
                # print(index)
                #总数
                total = total+1
                print("----当前为第 "+str(total)+" 文件-------------------------")
                #处理图片
                img = np.array([img_to_array(img)],dtype='float')/255.0
                #预测图片，获得预测结果标签
                preds = model.predict(img).tolist()[0]
                label = classes_id()[preds.index(max(preds))]
                confidence = max(preds)
                pre_class = label
                #获得图片路径
                # print(img)
                img_name = os.listdir(every_path)[index]
                img_ori_path = os.path.join(every_path,img_name)
                # real_class = img_name.split('_',1)[0]
                real_class = img_ori_path.split('\\')[-1].split('_')[0]
                # print(img_name)
                if(real_class == pre_class):
                    i = i + 1
                try:
                    with open(log_path,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                        file.write(img_ori_path)
                        file.write('\r\t')
                        file.write(real_class)
                        file.write('\r\n')
                        file.write(pre_class)
                        file.write('\r\n')
                except IOError as identifier:
                    print("error")
                else:
                    # print(log_path)
                    pass
    try:
        with open(log_path,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write('\r\t')
            file.write('\r\n')
            file.write('\r\n')
            file.write(str(i/total*100))
            file.write('\r\n')
    except IOError as identifier:
        print("error")
    else:
        print("finish")

if __name__ == "__main__":
    predict(model_path,test_path)