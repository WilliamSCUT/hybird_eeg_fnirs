import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as scio


######1、按照mark划分trail； 2、为trail标记任务；3、计算每个trail的血氧数值#############

def getHb_HbO2(data760, data850, SD, epsilon=[[1.6745, 0.5495], [0.7861, 1.1596]], DPF=[6, 6]):
    '''
    根据原始光信号的变化，计算出对应的血红蛋白浓度变化情况
    :param data760: 对应波长为760nm的出射光强值，表头是相对于的S-D光通道
    :param data850: 对应波长为850nm的出射光强值，表头是相对于的S-D光通道
    :param SD:      对应的是每个S-D光通道的距离
    :param epsilon: 760和850nm波长的光的吸光谱，每一行对应一个光波长，两列对应的是Hb，HbO2的吸光谱
    :param DPF:     差分路径因子，修正系数，一般取[6,6]做默认值，可查找相关论文得到更精准的值
    :return:        输出为血红蛋白浓度的信号值三维矩阵，第一维表示数量，第二维对应的每个S-D通道，第三维是[Hb, HbO2,tHb]的相对浓度
    '''

    length = data760.shape[0]  # 采样点数
    SD_num = SD.shape[0]  # 通道数
    Hb_HbO2 = np.zeros([length, SD_num, 3])
    # 计算参数A
    tDPF = np.array([DPF, DPF]).T
    K = epsilon * tDPF
    A = np.dot(np.linalg.inv(np.dot(K.T, K)), K.T) * 1000

    # 计算结果
    for m in range(SD_num):
        light_intensity = np.vstack([data760[:, m], data850[:, m]])
        OD = np.log(light_intensity.T)
        Hb_HbO2[:, m, :2] = np.dot(OD, A) / SD[m]

    Hb_HbO2[:, :, 2] = Hb_HbO2[:, :, 0] + Hb_HbO2[:, :, 1]
    Hb_HbO2[:, :, :] = Hb_HbO2[:, :, :] - Hb_HbO2[0, :, :]

    return Hb_HbO2


if __name__ == '__main__':
    num = 29
    # processed_data_path='./processed_data2/'
    processed_data_path = r'D:\dataset\Open_Access_Dataset_for_EEG_NIRS_Single-Trial_Classification\EEG-fNIRS' + '/'
    data_path = 'cnt.mat'
    mark_path = 'mrk.mat'
    mnt_path = 'mnt.mat'
    sampling_rate = 10
    channel_num = 36

    for n in range(1, num + 1):
        sub = str(n) if n >= 10 else '0' + str(n)
        # data=scio.loadmat(r'./data/subject '+sub+'/'+data_path)
        # mark=scio.loadmat(r'./data/subject '+sub+'/'+mark_path)
        # mnt=scio.loadmat(r'./data/subject '+sub+'/'+mnt_path)
        raw_data_path = r'D:\dataset\Open_Access_Dataset_for_EEG_NIRS_Single-Trial_Classification\EEG-fNIRS\subject '
        data_ = scio.loadmat(raw_data_path + sub + '\\' + data_path)
        mark_ = scio.loadmat(raw_data_path + sub + '\\' + mark_path)
        mnt_ = scio.loadmat(raw_data_path + sub + '\\' + mnt_path)
        data = data_['cnt']
        mark = mark_['mrk']
        mnt = mnt_['mnt']
        # cell_num=data.size
        cell_num = [1, 3, 5]

        SD = np.zeros([channel_num], dtype=np.float32)
        distance_3d = mnt['pos_3d'][0][0]
        for c in range(distance_3d.shape[1]):
            tmp = distance_3d[:, c]
            SD[c] = np.linalg.norm(tmp)

        processed_data = []
        processed_label = []
        # for i in range(cell_num):
        #     label=1
        #     if i%2==1:
        #         label=0
        #     tmp_data=data[0][i]['x'][0][0]
        #     tmp_mark=mark[0][i]['time'][0][0]
        #     trail_num=tmp_mark.shape[1]
        #     for tn in range(trail_num):
        #         trail_start=tmp_mark[0][tn]
        #         trail_start=trail_start/1000+2
        #         trail_end=trail_start+10
        #         trail_start=int(trail_start*sampling_rate)
        #         trail_end=int(trail_end*sampling_rate)
        #         trail_data=tmp_data[trail_start:trail_end][:]
        #         length1=trail_data.shape[0]
        #         data760 = np.zeros([length1, channel_num])
        #         data850=np.zeros([length1, channel_num])
        #
        #         data760[:,:channel_num]=trail_data[:,:channel_num]
        #         data850[:,:channel_num]=trail_data[:,channel_num:]
        #         Puo = getHb_HbO2(data760, data850, SD, epsilon =[[1.5458, 0.5495], [0.8399, 0.8653]], DPF = [6, 6] )
        #         processed_data.append(Puo)
        #         processed_label.append(label)
        # processed_data=np.stack(processed_data)
        # processed_label=np.stack(processed_label)
        # save_path=processed_data_path+'subject_'+str(n)+'/'
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # scio.savemat(save_path+'processed_data_sub'+str(n)+'.mat',{'data':processed_data,'label':processed_label})

        for i in cell_num:  # [1, 3, 5]
            tmp_data = data[0][i]['x'][0][0]  # (7180,72)
            tmp_mark = mark[0][i]['time'][0][0]  # (1,20)
            trail_num = tmp_mark.shape[1]  # 20
            for tn in range(trail_num):
                trail_start = tmp_mark[0][tn]
                trail_MA_start = trail_start / 1000 + 2
                trail_MA_end = trail_MA_start + 10
                trail_MA_start = int(trail_MA_start * sampling_rate)
                trail_MA_end = int(trail_MA_end * sampling_rate)
                trail_data = tmp_data[trail_MA_start:trail_MA_end][:]
                length1 = trail_data.shape[0]
                data760 = np.zeros([length1, channel_num])
                data850 = np.zeros([length1, channel_num])

                data760[:, :channel_num] = trail_data[:, :channel_num]
                data850[:, :channel_num] = trail_data[:, channel_num:]
                Puo = getHb_HbO2(data760, data850, SD, epsilon=[[1.5458, 0.5495], [0.8399, 0.8653]], DPF=[6, 6])
                processed_data.append(Puo)
                label = 1
                processed_label.append(label)
                print(tn)
            for tn in range(trail_num):
                trail_start = tmp_mark[0][tn]
                trail_rest_start = trail_start / 1000 + 13
                trail_rest_end = trail_rest_start + 10
                trail_rest_start = int(trail_rest_start * sampling_rate)
                trail_rest_end = int(trail_rest_end * sampling_rate)
                trail_data = tmp_data[trail_rest_start:trail_rest_end][:]
                length1 = trail_data.shape[0]
                data760 = np.zeros([length1, channel_num])
                data850 = np.zeros([length1, channel_num])

                data760[:, :channel_num] = trail_data[:, :channel_num]
                data850[:, :channel_num] = trail_data[:, channel_num:]
                Puo = getHb_HbO2(data760, data850, SD, epsilon=[[1.5458, 0.5495], [0.8399, 0.8653]], DPF=[6, 6])
                processed_data.append(Puo)
                label = 0
                processed_label.append(label)
        processed_data = np.stack(processed_data)
        processed_label = np.stack(processed_label)
        save_path = processed_data_path + 'subject_' + str(n) + '/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        scio.savemat(save_path + 'processed_data_sub' + str(n) + '.mat',
                     {'data': processed_data, 'label': processed_label})

        # DataT=data1['a']
        # distances=datadistances['d']
        # length1 = DataT.shape[0]
        # data760=np.zeros([length1, 40])
        # data850=np.zeros([length1, 40])
        # SD=np.zeros([40])

    #     for i in range(40):
    #         data760[:,i]=DataT[:,i]
    #         data850[:,i]=DataT[:,i+40]
    #         SD[i]=distances[i]
    #     Puo = getHb_HbO2(data760, data850, SD, epsilon =[[1.5458, 0.5495], [0.7861, 1.1596]], DPF = [6, 6] )
    #
    #     #导出到excel
    #     datatmp1= pd.DataFrame(Puo[:, :, 0]) #数组转为dataframe结构
    #     datatmp2= pd.DataFrame(Puo[:, :, 1])
    #     datatmp3= pd.DataFrame(Puo[:, :, 2])
    #
    #     writer = pd.ExcelWriter(r'D:\data\before\before'+f[n]+'.xlsx')
    #     datatmp1.to_excel(writer,'page_1',float_format='%.5f')
    #     datatmp2.to_excel(writer,'page_2',float_format='%.5f')
    #     datatmp3.to_excel(writer,'page_3',float_format='%.5f')
    #     writer.save()
    #
    # writer.close()