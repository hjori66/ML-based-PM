import csv
import pandas as pd
import torch
import numpy as np
import matplotlib
from scipy.signal import savgol_filter

dir_name = 'Cathode potential raw data'
data_address = dir_name + '\\' + 'data.csv'
train_address = dir_name + '\\' + 'train.pickle'
test_address = dir_name + '\\' + 'test.pickle'
file_names = ['Cathode potential_Ce0.1(100 to 300)', 'Cathode potential_Ce1(0.1 to 0.25)',
              'Cathode potential_Ce10(less than 0.01)', 'Cathode potential_Gd0.25(300 to 1200)',
              'Cathode potential_Gd2.5(0.4 to 0.6)', 'Cathode potential_Gd25(0.02 to 0.05)',
              'Cathode potential_La10(no codeposit)']

cathode_surface_address = dir_name + '\\' + 'Cathode_surface_area_'

# label_dict = {'Ce0.1': 0, 'Ce1': 1, 'Ce10': 2, 'Gd0.25': 3, 'Gd2.5': 4, 'Gd25': 5, 'La10': 6}
label_dict = {'Ce0.1': 0, 'Ce1': 0, 'Ce10': 0, 'Gd0.25': 0, 'Gd2.5': 0, 'Gd25': 0, 'La10': 1}
# label_dict = {'Ce0.1': 0, 'La10': 1}

volt_dict = ['100', '150', '200', '250', '300', '350', '400', '450', '500']
# volt_dict = ['100', '500']


def csv_to_data(slicer=6000):
    """
    csv data(실험값, Appendix 1 이론값)를 받아서 (Input, Label) 쌍의 List를 만들어 pkl 파일에 저장합니다.
    :param slicer: Slice the data ('slicer' timesteps)
    :return:
    """
    df_list = list()
    # 실험값 csv 읽어서 저장.
    for i, fname in enumerate(file_names):
        address = dir_name + '\\' + fname + '.csv'
        label_suffix = fname.split('_')[1]
        label = label_suffix.split('(')[0]

        data = pd.read_csv(address, header=[0, 1])
        col_list = list()
        for i, col_tuple in enumerate(data.columns):
            if i % 2 == 0:
                col_list.append(label + '_' + col_tuple[0])
                del data[col_tuple]
        data.columns = col_list
        df_list.append(data)
    data = pd.concat(df_list, axis=1)

    train_data = list()
    test_data = list()
    for col in data.columns:
        col_ = col.split('_')
        label = col_[0]
        volt = col_[1]
        if volt[-1] == 'A':
            volt = volt[:-3]
        tag = col_[2]
        slicer = min(slicer, len(data[col]))

        """
        input1 : 실험값
        input2 : Appendix 1 이론값
        input3 : (실험값, Appendix 1 이론값)
        """

        volt_data = pd.read_csv(cathode_surface_address + label + '.csv')
        input1 = np.array(data[col][:slicer])

        # savgol_filter를 이용한 smoothing.
        savgol_window = 5
        savgol_poly_deg = 3

        input1_list = data[col][:slicer].to_list()
        input1_smoothing = savgol_filter(input1_list, savgol_window, savgol_poly_deg)
        input2 = np.array(volt_data[volt][:slicer])
        # input3 = np.array([[float(input1_list[i]), float(input2[i])] for i in range(len(input1))])
        input3 = np.array([[float(input1_smoothing[i]), float(input2[i])] for i in range(len(input1))])
        if not label in label_dict:
            continue
        elif tag != '1' and volt in volt_dict:
            train_data.append((input3, label_dict[label]))
        elif tag == '1' and volt in volt_dict:
            test_data.append((input3, label_dict[label]))

    train_df = pd.DataFrame(train_data)[1:]
    test_df = pd.DataFrame(test_data)[1:]
    train_df.columns = ['input', 'label']
    test_df.columns = ['input', 'label']
    # data.to_csv(data_address)
    train_df.to_pickle(train_address + '_' + str(slicer))
    test_df.to_pickle(test_address + '_' + str(slicer))
    return train_df, test_df


def load_df(slicer=6000):
    """
    pkl 파일을 불러와서 Training dataset과 Test dataset을 반환합니다.
    :param slicer:
    :return: Training/Test dataset(Numpy Array)
    """
    train_df = pd.read_pickle(train_address + '_' + str(slicer))
    test_df = pd.read_pickle(test_address + '_' + str(slicer))
    return train_df, test_df
