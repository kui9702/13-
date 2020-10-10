'''
将文件夹里的文件分成多少组
'''

import os
import random
import shutil

def class_file(img_path,folder_path,number):
    img_number = 100
    img_list = os.listdir(img_path)
    print(img_list)
    random.shuffle(img_list)
    print('-------------------------------')
    # print(img_list)
    for i in range(number):
        for j in range(img_number):
            j = j + i * img_number
            if j < len(img_list):
                every_path = os.path.join(folder_path,str(i))
                print(every_path)
                if not os.path.exists(every_path):
                    os.makedirs(every_path)
                shutil.copyfile(os.path.join(img_path,img_list[j]),os.path.join(every_path,img_list[j]))
    print('finish')
if __name__ == "__main__":
    num = 28 #文件夹数量
    img_path = r'G:\13classes_transfer\new_data2test\test_img'   #图片根目录
    folder_path = r'G:\13classes_transfer\new_data2test\28zu'    #文件夹根目录
    class_file(img_path,folder_path,num)