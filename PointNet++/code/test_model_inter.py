import torch
from torch import nn
import pandas as pd
import numpy as np
import pointnet2_cls_ssg as pointnet2
from dataset import pointdata,norm_data
from torch.utils.data import Dataset,DataLoader
import pickle
import os
import datetime
import matplotlib as mpl
mpl.use("WebAgg")
import pylab as pl


if not os.path.exists("inter_datainfo.pkl"):
    min_val,max_val, num_data,num_points,in_channel = norm_data(r'../inter-input1')
    label = pd.read_csv(r'../inter-output/output.csv').values
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


model = pointnet2.get_model(out_channel = 2,in_channel=data_info['in_channel'])
device = torch.device('cuda:0')
model.load_state_dict(torch.load('train_model_inter/model_weights_16_0.04825_0.17415.pth'))
model.to(device)

input_path = r'../inter-input1'
output_path_test = r'../inter-output/output_test.csv'
point_number = 5000
input_preStr = 'inter'

data_valid = pointdata(input_path,output_path_test,data_info,point_number,input_preStr)
dl_valid = DataLoader(data_valid,batch_size =12 ,shuffle = False,num_workers=2)

pred_y = []
real_y = []
for i,(x,y) in enumerate(dl_valid):
    if i>10:
        break

    py = model(x.float().cuda())
    pred_y.append(py.detach().cpu())
    real_y.append(y)

pred_y = torch.cat(pred_y).numpy()
real_y = torch.cat(real_y).numpy()

pred_y = (data_info['output_max'] - data_info['output_min'])*pred_y+data_info['output_min']
real_y = (data_info['output_max'] - data_info['output_min'])*real_y+data_info['output_min']

mape = np.abs(np.mean(pred_y,axis=0) - np.mean(real_y,axis=0) )/np.std(real_y,axis=0) 

pl.figure()
pl.close("all")
pl.plot(real_y[:,0],'r.:')
pl.plot(np.mean(real_y[:,0])*np.ones(real_y.shape[0]),'r-')
pl.plot(pred_y[:,0],'b.:')
pl.plot(np.mean(pred_y[:,0])*np.ones(pred_y.shape[0]),'b-')
pl.title("v_hf_bsurf (mape = %0.4f)"%mape[0])
pl.legend(["real val","real val(mean)","pred val","pred val(mean)"])
pl.xlabel("step")
pl.ylabel("mape")

pl.figure()
pl.plot(real_y[:,1],'r.:')
pl.plot(np.mean(real_y[:,1])*np.ones(real_y.shape[0]),'r-')
pl.plot(pred_y[:,1],'b.:')
pl.plot(np.mean(pred_y[:,1])*np.ones(pred_y.shape[0]),'b-')

pl.title("v_hf_bsolv (mape = %0.4f)"%mape[1])
pl.legend(["real val","real val(mean)","pred val","pred val(mean)"])
pl.xlabel("step")
pl.ylabel("mape")
pl.show()