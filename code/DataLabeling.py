import os
import re
import csv
import pandas as pd
import Const
import pydicom as dicom
import glob

XML_READ_FILE_PATH = os.path.join(Const.MAIN_PATH, Const.DATA_PATH,"ori_data.xlsx")
CSV_ALL_DATA = os.path.join(Const.MAIN_PATH, Const.DATA_PATH, "sort_all_data.csv")
CSV_WRITE_FILE_PATH = os.path.join(Const.MAIN_PATH, Const.DATA_PATH,"labeling_data.csv")
DATA_FILE_PATH = os.path.join(Const.MAIN_PATH, Const.DATA_PATH,"train")
#
# XML_READ_FILE_PATH = "data\stage_2_train.csv"
# XML_WRITE_FILE_PATH = "data\labeling_data_all.xlsx"

id_list = []
label_list = []


# def getClassificationLabel():
#     # data_only=True로 해야 수식이 아닌 값으로 불러옴
#     load_wb = openpyxl.load_workbook(XML_READ_FILE_PATH, data_only=True)
#     load_ws = load_wb['Sheet1']
#
#     index = 0
#     while True:
#         # 새로운 환자 데이터 id의 시작 index
#         item_index = 2 + (6 * index)
#         id = load_ws.cell(item_index, 1).value
#
#         if not id:
#             break
#         else:
#             id = id.split("_")[1]
#         label = ""
#         any = load_ws.cell(item_index, 2).value
#         print(f"save id = {id} any = {any}")
#
#         # 출혈이 없는 경우
#         if any == 0:
#             label = "nomal"
#             print(f"id = {id} label = {label}")
#         else:
#             # 어디선가 출혈 부위가 있는 경우 찾기
#             for i in range(1, 6):
#                 bleeder_index = 2 + (6 * index) + i
#                 part = load_ws.cell(bleeder_index, 1).value.split("_")[2]
#                 value = load_ws.cell(bleeder_index, 2).value
#                 if value == 1:
#                     label = part
#                     print(f"save id = {id} label = {label}")
#                     break
#         # 만약 전부 찾았는데 출혈 부위를 못찾은 경우
#         if not label:
#             print("not find label")
#         else:
#             id_list.append(id)
#             label_list.append(label)
#         index += 1

# Label CSV파일 기준으로 있는 데이터만 새로저장
def restoreData():
    csv_data = pd.read_csv(CSV_ALL_DATA, header=None)
    # last_data_id = re.compile("_\w+.").search(os.listdir(Const.DATA_ALL_PATH).pop()).group()[1:11]

    label = ""
    index = 0
    try:
        while True:
            item_index = 1 + (6 * index)
            id = csv_data[0][item_index].split("_")[1]
            if id:
            # if os.path.isfile(f"{Const.DATA_ALL_PATH}ID_{id}.dcm"):
                print(f"ID is {id}")
                any = csv_data[1][item_index]
                print(any)
                if any == "0" or any == 0:
                    label = "nomal"
                    print(f"id = {id} label = {label}")
                else:
                    print("bleading")
                    # 어디선가 출혈 부위가 있는 경우 찾기
                    for i in range(1, 6):
                        bleeder_index = 1 + (6 * index) + i
                        part = csv_data[0][bleeder_index].split("_")[2]
                        value = csv_data[1][bleeder_index]
                        if value == "1" or value == 1:
                            label = part
                            print(f"Bleeder id = {id} label = {label}")
                            break
                # 만약 전부 찾았는데 출혈 부위를 못찾은 경우
                if not label:
                    print("not find label")
                else:
                    id_list.append(id)
                    label_list.append(label)
            else :
                break;

            # if last_data_id == id:
            #     print("FIND LAST ID")
            index += 1

    except Exception as e:
        print(e)

# CSV파일로 정보를 내보내느 메소드
def saveCSV():
    if os.path.isfile(CSV_WRITE_FILE_PATH):
        os.remove(CSV_WRITE_FILE_PATH)
    label_df = pd.DataFrame({'id': id_list, 'label': label_list})
    print(label_df)
    label_df.to_csv(
        CSV_WRITE_FILE_PATH,
        index=False
    )

#전체데이터 sorting
def allDataCSVSorting():
    all_data = pd.read_csv(CSV_ALL_DATA)
    all_data = all_data.sort_values("ID", ascending=True)
    # all_data = pd.read_csv(CSV_ALL_DATA, header=None)
    print(all_data)
    all_data.to_csv(CSV_ALL_DATA,index=False)

# DCM 정보를 확인하는 메소드
def confirmDCMInfo():
    data_path = os.path.join(Const.DATA_ALL_PATH)
    file_list = os.listdir(data_path)
    for file in file_list:
        dcm = dicom.dcmread(os.path.join(data_path, file))
        print(dcm.ImagePositionPatient)

# DCM파일과 기본 LabelCSV파일의 정보를 묶어 다시 CSV로 내보내는 메소드
def convertDCMInfoCSV():
    all_data = pd.read_csv(CSV_ALL_DATA, header=None, low_memory=False)
    index = 0
    id_index = 1
    id_list
    any_list = []
    epidural_list = []
    intraparenchymal_list = []
    intraventricular_list = []
    subarachnoid_list = []
    subdural_list = []
    SOPInstantUID_list = []
    PatientID_list = []
    ImagePosision_list = []
    while True:
        try:
            dcm_id = ("_").join(str(all_data[0][id_index]).split("_")[:2])
            print(dcm_id)
            dcm = dicom.dcmread(os.path.join(Const.MAIN_PATH,Const.DATA_PATH,"stage_2_train",f"{dcm_id}.dcm"))
            id_list.append(dcm_id)
            any_list.append(all_data[1][id_index])
            epidural_list.append(all_data[1][id_index+1])
            intraparenchymal_list.append(all_data[1][id_index+2])
            intraventricular_list.append(all_data[1][id_index+3])
            subarachnoid_list.append(all_data[1][id_index+4])
            subdural_list.append(all_data[1][id_index+5])
            SOPInstantUID_list.append(dcm.SOPInstanceUID)
            print(dcm.SOPInstanceUID)
            PatientID_list.append(dcm.PatientID)
            print(dcm.PatientID)
            ImagePosision_list.append(dcm.ImagePositionPatient[2])
            print(dcm.ImagePositionPatient[2])
        except KeyError:
            break
        except Exception as e:
            print(e)
        index += 1
        id_index += 6
    label_df = pd.DataFrame({
        'ID': id_list,
        'any': any_list,
        'epidural': epidural_list,
        'intraparenchymal': intraparenchymal_list,
        'intraventricular': intraventricular_list,
        'subarachnoid': subarachnoid_list,
        'subdural': subdural_list,
        'SOPInstantUID': SOPInstantUID_list,
        'PatientID': PatientID_list,
        'ImagePosision': ImagePosision_list
    })
    label_df.to_csv(
        os.path.join(Const.MAIN_PATH,Const.DATA_PATH, "data_patient.csv"),
        index=False
    )

# confirmDCMInfo()
convertDCMInfoCSV()
#allDataCSVSorting()
# restoreData()
# saveCSV()
# getClassificationLabel()
# saveXlsx()
