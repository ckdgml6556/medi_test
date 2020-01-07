import os
import csv
import shutil
import mritopng
import glob
import Const
import pydicom as dicom
import numpy as np
import png
import PIL
import re
import pandas as pd

#PNG파일 Option
PNG = False

label_dict = ["nomal", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
def renameFlie():
    file_list = glob.glob(f"{Const.DATA_ALL_PATH}\\*.dcm")
    for file in file_list:
        new_file = file.split("\\").pop()[3:]
        print(new_file)
        os.rename(file, f"{Const.DATA_ALL_PATH}\\{new_file}")


# 전체 폴더에서 분류
def classficationFile():
    rdf = csv.reader(open(Const.DATA_LABEL_CSV, 'r', encoding='utf-8'))
    for line in rdf:
        file_name = f"ID_{line[0]}.dcm"
        if os.path.isfile(f"{Const.DATA_ALL_PATH}\\{file_name}"):
            dir_path = f"{Const.DATA_ALL_PATH}\\{line[1]}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"make dir {dir_path}")

            print(dir_path)
            shutil.move(f"{Const.DATA_ALL_PATH}\\{file_name}", f"{dir_path}\\")


# 일정 비율 만큼 전체데이터에서 트레인과 테스트 셋 파일 복사
def seperateData(train_ratio, val_ratio, test_ratio):
    for label in label_dict:
        all_path = f"{Const.DATA_ALL_PATH}\\{label}"
        print(label)
        if Const.CURRENT_TYPE == Const.TYPE_NOMAL:
            new_label = label if label =="nomal" else "abnomal"
            train_path = f"{Const.DATA_TRAIN_PATH}\\{new_label}"
            val_path = f"{Const.DATA_VAL_PATH}\\{new_label}"
            test_path = f"{Const.DATA_TEST_PATH}\\{new_label}"
        else:
            train_path = f"{Const.DATA_TRAIN_PATH}\\{label}"
            val_path = f"{Const.DATA_VAL_PATH}\\{label}"
            test_path = f"{Const.DATA_TEST_PATH}\\{label}"

        print(all_path)

        if not os.path.exists(train_path):
            os.makedirs(train_path)
            print(f"make dir {train_path}")

        if not os.path.exists(val_path):
            os.makedirs(val_path)
            print(f"make dir {val_path}")

        if not os.path.exists(test_path):
            os.makedirs(test_path)
            print(f"make dir {test_path}")

        file_list = glob.glob(f"{all_path}\\*")
        file_format = ".png" if PNG else ".jpg"
        file_list = [file for file in file_list if file.endswith(file_format)]
        # 전체 데이터 쓰기
        # file_count = len(file_list)
        # print(file_count)
        #정해진 데이터 쓰기
        max_count = 20000
        file_count = max_count if len(file_list) > max_count else len(file_list)
        train_count = int(round(file_count * train_ratio))
        val_count = int(round(file_count * val_ratio))

        print(train_count)
        print(val_count)
        train_list = file_list[0:train_count]
        val_list = file_list[train_count:train_count + val_count]
        test_list = file_list[train_count + val_count:file_count]
        for train_file in train_list:
            train_file = train_file.split("\\").pop()
            if not os.path.isfile(f"{train_path}\\{train_file}"):
                shutil.copy(f"{all_path}\\{train_file}", f"{train_path}\\{label}_{train_file}")

        for val_file in val_list:
            val_file = val_file.split("\\").pop()
            if not os.path.isfile(f"{val_path}\\{val_file}"):
                shutil.copy(f"{all_path}\\{val_file}", f"{val_path}\\{label}_{val_file}")

        for test_file in test_list:
            test_file = test_file.split("\\").pop()
            if not os.path.isfile(f"{test_path}\\{test_file}"):
                shutil.copy(f"{all_path}\\{test_file}", f"{test_path}\\{label}_{test_file}")

    # data_frame = pd.read_csv(DATA_LABEL_CSV, header=None)
    # data_id = data_frame[0][1:].to_list()
    # data_label = data_frame[1][1:].to_list()



def dicomToJpg():
    dir_list = os.listdir(f"{Const.DATA_ALL_PATH}\\")
    print(dir_list)
    for dir in dir_list:
        file_list = os.listdir(f"{Const.DATA_ALL_PATH}\\{dir}")
        for file in file_list:
            ds = dicom.dcmread(f"{Const.DATA_ALL_PATH}\\{dir}\\{file}")
            print(ds)
            # print(type(ds.WindowCenter))
            # if str(type(ds.WindowCenter)) == "<class 'pydicom.multival.MultiValue'>":
            #     wc = float(ds.WindowCenter[0])
            #     ww =  float(ds.WindowWidth[0])
            # else :
            #     wc = ds.WindowCenter
            #     ww = ds.WindowWidth
            # img = ds.pixel_array
            # arr = img * ds.RescaleSlope + ds.RescaleIntercept
            # # min = int(wc) - (int(ww) * 2)
            # # max = int(wc) + (int(ww) / 2)
            # # arr[arr < min] = min
            # # arr[arr > max] = max
            # scaled_img = cv2.convertScaleAbs(arr, beta= (255.0 / ww))
            # # # cv2.imshow('sample image dicom',ds.pixel_array)
            # # # print(ds)
            # # # pixel_array_numpy = ds.pixel_array
            # if not PNG:
            #     image = file.replace('.dcm', '.jpg')
            # else:
            #     image = file.replace('.dcm', '.png')
            # cv2.imwrite(os.path.join(f"{Const.DATA_ALL_PATH}\\{dir}", image), scaled_img)


def moveJPG():
    dir_list = os.listdir(Const.DATA_ALL_PATH)
    for dir in dir_list:
        class_dir_path = os.path.join(Const.DATA_ALL_PATH, dir)
        sub_dir_list = os.listdir(class_dir_path)
        index = 1
        for sub_dir in sub_dir_list:
            sub_dir_path = os.path.join(class_dir_path, sub_dir)
            file_list = os.listdir(sub_dir_path)
            for file in file_list:
                file = os.path.join(sub_dir_path,file)
                shutil.move(file, f"{class_dir_path}\\{index}.jpg")
                index += 1
            os.rmdir(sub_dir_path)
#
# def dicomToJpg():
#     path = "D:\\data\\"
#     file = "ID_0c6ee8b7a.dcm"
#     ds = dicom.dcmread(f"{path}{file}")
#     img = ds.pixel_array
#     scaled_img = cv2.convertScaleAbs(img, alpha=(255.0 / 80))
#     # cv2.imshow('sample image dicom',ds.pixel_array)
#     #print(ds)
#     # pixel_array_numpy = ds.pixel_array
#     if not PNG:
#         image = file.replace('.dcm', '.jpg')
#     else:
#         image = file.replace('.dcm', '.png')
#     cv2.imwrite(os.path.join(path, image), scaled_img)

def deleteDCMFiles():
    dir_list = glob.glob(f"{Const.DATA_ALL_PATH}\\*")
    for dir in dir_list:
        file_list = glob.glob(f"{dir}\\*.dcm")
        for file in file_list:
            os.remove(file)
            print(f"remove file {file}")

def changeDCM():
    dir_list = os.listdir(f"{Const.DATA_ALL_PATH}\\")
    print(dir_list)
    for dir in dir_list:
        file_list = os.listdir(f"{Const.DATA_ALL_PATH}\\{dir}")
        for file in file_list:
            new_file = file.split(".dcm")[0]
            os.rename(f"{Const.DATA_ALL_PATH}\\{dir}\\{file}", f"{Const.DATA_ALL_PATH}\\{dir}\\{new_file}.png")


# renameFlie()
# deleteDCMFiles()
# dicomToJpg()
# moveJPG()
# deleteDCMFiles()
seperateData(Const.TRAIN_BIAS, Const.VAL_BIAS, Const.TEST_BIAS)
# changeDCM()
# classficationFile()
# dicomToJpg()