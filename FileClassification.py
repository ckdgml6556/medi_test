import os
import csv
import shutil
import mritopng
import glob
import Const
import pandas as pd

DATA_LABEL_CSV = "data\\labeling_data.csv"

label_dict = ["nomal","epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]

def changFileName():
    for file in glob.glob(f"{Const.DATA_ALL_PATH}\\*.dcm"):
        file_name = file.split("_")[1]
        os.rename(file, f"{Const.DATA_ALL_PATH}\\{file_name}")

# 전체 폴더에서 분류
def classficationFile():
    rdf = csv.reader(open(DATA_LABEL_CSV, 'r', encoding='utf-8'))
    for line in rdf:
        file_name = f"{line[0]}.dcm"
        if os.path.isfile(f"{Const.DATA_ALL_PATH}\\data\\{file_name}"):
            dir_path = f"{Const.DATA_ALL_PATH}\\{line[1]}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"make dir {dir_path}")

            print(dir_path)
            shutil.move(f"{Const.DATA_ALL_PATH}\\{file_name}", f"{dir_path}\\")

# 일정 비율 만큼 전체데이터에서 트레인과 테스트 셋 파일 복사
def seperateData(train_ratio, val_ratio, test_ratio):
    for label in label_dict :
        print(label)
        all_path = f"{Const.DATA_ALL_PATH}\\{label}"
        train_path = f"{Const.DATA_TRAIN_PATH}\\{label}"
        val_path = f"{Const.DATA_VAL_PATH}\\{label}"
        test_path = f"{Const.DATA_TEST_PATH}\\{label}"

        print(all_path)

        if not os.path.exists(train_path) :
            os.makedirs(train_path)
            print(f"make dir {train_path}")

        if not os.path.exists(val_path):
            os.makedirs(val_path)
            print(f"make dir {val_path}")

        if not os.path.exists(test_path):
            os.makedirs(test_path)
            print(f"make dir {test_path}")

        file_list = glob.glob(f"{all_path}\\*")
        file_list = [file for file in file_list if file.endswith(".png")]
        file_count = len(file_list)
        print(file_count)
        train_count = int(round(file_count * train_ratio))
        val_count = int(round(file_count * val_ratio))


        train_list = file_list[0:train_count]
        val_list = file_list[train_count:train_count+val_count]
        test_list = file_list[train_count+val_count:]

        for train_file in train_list:
            train_file = train_file.split("\\")[3]
            if not os.path.isfile(f"{train_path}\\{train_file}"):
                shutil.copy(f"{all_path}\\{train_file}", f"{train_path}\\{train_file}")

        for val_file in val_list:
            val_file = val_file.split("\\")[3]
            if not os.path.isfile(f"{val_path}\\{val_file}"):
                shutil.copy(f"{all_path}\\{val_file}", f"{val_path}\\{val_file}")

        for test_file in test_list:
            test_file = test_file.split("\\")[3]
            if not os.path.isfile(f"{test_path}\\{test_file}"):
                shutil.copy(f"{all_path}\\{test_file}", f"{test_path}\\{test_file}")



    # data_frame = pd.read_csv(DATA_LABEL_CSV, header=None)
    # data_id = data_frame[0][1:].to_list()
    # data_label = data_frame[1][1:].to_list()


#dicom파일을 png로바꾸기 라이브러리 다운 https://github.com/danishm/mritopng
def dicomToJpg():
    dir_list = os.listdir(f"{Const.DATA_ALL_PATH}\\")
    print(dir_list)
    for dir in dir_list:
        file_list = os.listdir(f"{Const.DATA_ALL_PATH}\\{dir}")
        for file in file_list:
            mritopng.convert_file(f"{Const.DATA_ALL_PATH}\\{dir}\\{file}", f"{Const.DATA_ALL_PATH}\\{dir}\\{file}.png")


changFileName()
# dicomToJpg()
# seperateData(0.6, 0.2, 0.2)
#classficationFile()

