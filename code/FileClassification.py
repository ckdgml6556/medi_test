import os
import csv
import shutil
import cv2
import mritopng
import glob
import Const
import pydicom as dicom
from pydicom.filebase import DicomBytesIO
import numpy as np
import png
import PIL
import re
import pandas as pd

# PNG파일 Option
PNG = False

label_dict = ["nomal", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


# def renameFlie():
#     file_list = glob.glob(f"{Const.DATA_ALL_PATH}\\*.dcm")
#     for file in file_list:
#         new_file = file.split("\\").pop()[3:]
#         print(new_file)
#         os.rename(file, f"{Const.DATA_ALL_PATH}\\{new_file}")
#

# 환자 아이디 별로 묶기
def collectPatientFile():
    file_list = glob.glob(f"{Const.DATA_ALL_PATH}\\*.dcm")
    index = 1
    for file in file_list:
        patient_path = f"{Const.DATA_PATIENT_PATH}{(index / 2000)}"
        if not os.path.exists(patient_path):
            os.makedirs(patient_path)
            print(f"make dir {patient_path}")
        ds = dicom.dcmread(file, force=True)
        if not os.path.exists(os.path.join(patient_path, ds.PatientID)):
            os.makedirs(os.path.join(patient_path, ds.PatientID))
            print(f"make dir {ds.PatientID}")
        shutil.copy(file, os.path.join(patient_path, ds.PatientID, file.split("\\").pop()))
        index += 1


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
            new_label = label if label == "nomal" else "abnomal"
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
        # 정해진 데이터 쓰기
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

# 이미지의 윈도우를 수정하는 메소드
def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

# 3가지의 window가 다른 사진을 하나로 합치는 메소드
def apply_window_policy(image):
    image1 = apply_window(image, 40, 80)  # brain
    image2 = apply_window(image, 80, 200)  # subdural
    image3 = apply_window(image, 40, 380)  # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1, 2, 0)
    return image

# 이미지 리스케일 메소드
def rescale_image(image, slope, intercept):
    return image * slope + intercept

# Dicom 파일을 삭제하는 메소드
def dicomToJpg():
    dir_list = os.listdir(os.path.join(Const.DATA_ALL_PATH))
    for dir in dir_list:
        file_list = os.listdir(os.path.join(Const.DATA_ALL_PATH, dir))
        for file in file_list:
            try:
                ds = dicom.dcmread(os.path.join(Const.DATA_ALL_PATH,dir,file), force=True)
                print(ds)
                rs = ds.RescaleSlope
                print(rs)
                ri = ds.RescaleIntercept
                if str(type(ds.WindowCenter)) == "<class 'pydicom.multival.MultiValue'>":
                    wc = float(ds.WindowCenter[0])
                    ww = float(ds.WindowWidth[0])
                else:
                    wc = ds.WindowCenter
                    ww = ds.WindowWidth
                print(f"wc = {wc}, ww = {ww}, rs = {rs}, ri = {ri}")
                img = ds.pixel_array
                img = rescale_image(img, rs, ri)
                window_image = apply_window_policy(img)
                window_image -= window_image.min((0, 1))
                window_image = (255 * window_image).astype(np.uint8)
                window_image = cv2.convertScaleAbs(window_image, beta=(255.0 / ww))
                if not PNG:
                    image = file.replace('.dcm', '.jpg')
                else:
                    image = file.replace('.dcm', '.png')
                cv2.imwrite(os.path.join(f"{Const.DATA_ALL_JPG_PATH}\\{dir}", image), window_image)
            except Exception as e:
                print(e)


#
# def moveJPG():
#     dir_list = os.listdir(Const.DATA_ALL_PATH)
#     for dir in dir_list:
#         if dir.find(".jpg"): continue
#         class_dir_path = os.path.join(Const.DATA_ALL_PATH, dir)
#         sub_dir_list = os.listdir(class_dir_path)
#         index = 1
#         for sub_dir in sub_dir_list:
#             sub_dir_path = os.path.join(class_dir_path, sub_dir)
#             file_list = os.listdir(sub_dir_path)
#             for file in file_list:
#                 file = os.path.join(sub_dir_path, file)
#                 shutil.move(file, os.path.join(class_dir_path, f"{index}.jpg"))
#                 index += 1
#             os.rmdir(sub_dir_path)


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


# Dicom파일을 삭제하는 메소드
def deleteDCMFiles():
    dir_list = glob.glob(f"{Const.DATA_ALL_PATH}\\*")
    for dir in dir_list:
        file_list = glob.glob(f"{dir}\\*.dcm")
        for file in file_list:
            os.remove(file)
            print(f"remove file {file}")


# def changeDCM():
#     dir_list = os.listdir(f"{Const.DATA_ALL_PATH}\\")
#     print(dir_list)
#     for dir in dir_list:
#         file_list = os.listdir(f"{Const.DATA_ALL_PATH}\\{dir}")
#         for file in file_list:
#             new_file = file.split(".dcm")[0]
#             os.rename(f"{Const.DATA_ALL_PATH}\\{dir}\\{file}", f"{Const.DATA_ALL_PATH}\\{dir}\\{new_file}.png")


# renameFlie()
# deleteDCMFiles()
# dicomToJpg()
# moveJPG()
# deleteDCMFiles()
# seperateData(Const.TRAIN_BIAS, Const.VAL_BIAS, Const.TEST_BIAS)
# changeDCM()
# classficationFile()
# dicomToJpg()
collectPatientFile()
