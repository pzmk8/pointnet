import torch
from torch import nn
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pickle

def norm_data(datapath):
    min_val = np.array([1e20,])
    max_val = np.array([-1e20,])
    num_data = 0
    num_points = []
    for f in os.listdir(datapath):
        if f[-3:] == 'csv':
            data = pd.read_csv(os.path.join(datapath,f)).values
            if min_val.shape[0] == 1:
                min_val = min_val.repeat(data.shape[1])
                max_val = max_val.repeat(data.shape[1])
                in_channel = data.shape[1]
            min_val = np.min([min_val,data.min(axis=0)],axis=0)
            max_val = np.max([max_val,data.max(axis=0)],axis=0)
            num_data += 1
            num_points.append(data.shape[0])
    return min_val,max_val, num_data,num_points,in_channel






class pointdata(Dataset):
    def __init__(self,input_path,output_path,data_info,point_number,input_preStr):
        self._input_path  = input_path
        self._label = pd.read_csv(output_path).values
        self._label[:,1:] = 2*(self._label[:,1:] - data_info['output_min']) / (data_info['output_max']- data_info['output_min']) - 1
        self._data_info = data_info
        self._input_preStr = input_preStr
        self.input_norm_d = data_info['input_max'] - data_info['input_min'] 
        self.point_number = point_number
        self._data_info['num_data'] = self._label.shape[0]


    def __len__(self):
        return self._data_info['num_data']
    
    def __getitem__(self,index):
        fileNo = int(self._label[index,0])
        data = 2*(pd.read_csv(os.path.join(self._input_path,'%s%d.csv'%(self._input_preStr,fileNo))).values - self._data_info['input_min'])/self.input_norm_d - 1
        label = self._label[index,1:]
        np.random.shuffle(data)
        return data[:self.point_number,:].T, label


if __name__ == '__main__':
    if not os.path.exists("inter_datainfo.pkl"):
        min_val,max_val, num_data,num_points,in_channel = norm_data(r'F:\problem\043\inter-input')
        label = pd.read_csv(r'F:\problem\043\inter-output\output.csv').values
        label_min = label[:,1:].min(axis=0)
        label_max = label[:,1:].max(axis=0)
        data_info = {'input_min':min_val,
            'input_max':max_val,
            'output_min':label_min,
            'output_max':label_max,
            'num_data':num_data,
            'num_points':num_points,
            'in_channel':in_channel}
        pickle.dump(data_info,open('inter_datainfo.pkl','wb'))
    else:
        data_info = pickle.load(open('inter_datainfo.pkl','rb'))

    input_path = r'F:\problem\043\inter-input'
    output_path = r'F:\problem\043\inter-output\output.csv'
    point_number = 5000
    input_preStr = 'inter'
    data = pointdata(input_path,output_path,data_info,point_number,input_preStr)