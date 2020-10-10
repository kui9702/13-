#结果输出
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
import shutil
from PIL import ImageFile
from datetime import datetime



# 解决读取截断图片问题，设为True
ImageFile.LOAD_TRUNCATED_IMAGES = True


#当前时间
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

normal_size = 299
#日志文件路径
log_path = r'G:/13classes_transfer/log'
#模型存储路径
model_path =  r'G:\13classes_transfer/ResNet50.keras/015ResNet50.keras'
#测试文件夹路径
test_path = r'G:\13classes_transfer\new_data2test\test_img'


if not os.path.exists(log_path):
    os.makedirs(log_path)

log_path = os.path.join(log_path,now + 'log.txt')

def classes_id():
    with open('./train_class_idx.txt','r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


def predict(model_path,path):
    #种类数量
    classNumber = 13
    model = load_model(model_path, compile=False) 
    for root,dirs,files in os.walk(path):
        for file in files:
            error_img = os.path.join(path,file)
            try:
                cv2.resize(cv2.imread(error_img),(normal_size,normal_size))
            except:
                os.remove(error_img)
                
    data_list = list(
        map(lambda x: cv2.resize(cv2.imread(os.path.join(path, x)),
                                        (normal_size, normal_size)), os.listdir(path)))

    i,total = 0,0
    for index, img in enumerate(data_list):
        total = total+1
        print("----当前为第 "+str(total)+" 文件-------------------------")
        img = np.array([img_to_array(img)],dtype='float')/255.0
        preds = model.predict(img).tolist()[0]

        top1_grade = np.sort(preds)[classNumber-1:classNumber][0]
        top1= np.argsort(preds)[classNumber-1:classNumber][0]
        top2_grade  = np.sort(preds)[classNumber-2:classNumber-1][0]
        top2 = np.argsort(preds)[classNumber-2:classNumber-1][0]
        print(top1)
        print(top1_grade)
        print(top2)
        print(top2_grade)

        pre_class_1 = classes_id()[top1]
        pre_class_2 = classes_id()[top2]
        confidence = max(preds)
        img_name = os.listdir(path)[index]
        img_ori_path = os.path.join(path,img_name)
        real_class = img_name.split('_',1)[1].split('_',1)[0]
        if(real_class == pre_class_1 or real_class == pre_class_2):
            i = i + 1
            shutil.copyfile(img_ori_path,os.path.join(r'G:\13classes_transfer\result',str(pre_class_1)+'_'+str(top1_grade)+'.'+str(img_ori_path.split('\\')[-1].split('.')[-1])))
            # shutil.copyfile(img_ori_path,os.path.join(r'G:\13classes_transfer\result\right',str(pre_class_1)+'_'+str(round(top1_grade,2))+'_'+str(pre_class_2)+'_'+str(round(top2_grade,2))+'===='+str(img_ori_path.split('\\')[-1])))
        else:
            shutil.copyfile(img_ori_path,os.path.join(r'G:\13classes_transfer\result',str(pre_class_1)+'_'+str(top1_grade)+'.'+str(img_ori_path.split('\\')[-1].split('.')[-1])))
            # shutil.copyfile(img_ori_path,os.path.join(r'G:\13classes_transfer\result\error',str(pre_class_1)+'_'+str(round(top1_grade,2))+'_'+str(pre_class_2)+'_'+str(round(top2_grade,2))+'===='+str(img_ori_path.split('\\')[-1])))
        try:
            with open(log_path,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                file.write(img_ori_path)
                file.write('\r\t')
                file.write(real_class)
                file.write('\r\n')
                file.write('top1:'+str(classes_id()[top1])+'_'+str(round(top1_grade,2))+ 'top2:'+str(classes_id()[top2])+'_'+str(round(top2_grade,2)))
                file.write('\r\n')
        except IOError as identifier:
            print("error")
        else:
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