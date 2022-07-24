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


def train_step(model,optimizer,x,labels):
    model.train()
    # 梯度清零
    optimizer.zero_grad()
    # 正向传播求损失
    # pdb.set_trace()
    x = x.float().cuda()
    labels = labels.float().cuda()
    predictions = model(x)

    loss,metric = model.loss_func(predictions,labels)
    
    # 反向传播求梯度
    loss.backward()
    optimizer.step()
    return loss.item(),metric.item()     #item()返回的是一个浮点型数据，

def valid_step(model,x,labels):
    model.eval()
    # pdb.set_trace()
    x = x.float().cuda()
    labels = labels.float().cuda()
    predictions = model(x)
    # labels = labels.cuda()
    loss,metric = model.loss_func(predictions,labels)
    
    return loss.item(),metric.item()


def train(model,dl_train,dl_valid,epochs = 100,start=1,lr=1e-4):
    dfhistory = pd.DataFrame(columns = ["epoch","mse",'mae',"val_mse","val_mae"]) 
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model.loss_func = pointnet2.get_loss()
    print("Start Training...")
    print("=========="*8 + "%s"%nowtime)
    log_step_freq = 10    #没实际意义，代表10
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(start,epochs+start):  
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (x,labels) in enumerate(dl_train, 1):
            loss,metric = train_step(model,optimizer,x,labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] mse: %.5f, mae: %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (x,labels) in enumerate(dl_valid, 1):
            if val_step>200:
                break
            val_loss,val_metric = valid_step(model,x,labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric
            # print(("[val step = %d] loss: %.3f, mAP: %.3f") % (val_step, val_loss, val_metric))

        # pdb.set_trace()
        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step,
                val_loss_sum/val_step, val_metric_sum/val_step)   #dfhistory = pd.DataFrame(columns = ["epoch","mse",'mae',"val_mse","val_mae"])
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print("\nEPOCH = %d, mse = %.5f, mae = %.5f, val_mse = %.5f, val_mae = %.5f"%info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
        if epoch % 2 == 0:
            torch.save(model.state_dict(),'train_model_bulk/model_weights_%d_%0.5f_%0.5f.pth'%(epoch,val_loss_sum/val_step,val_metric_sum/val_step))
    return dfhistory



if not os.path.exists("bulk_datainfo.pkl"):
    min_val,max_val, num_data,num_points,in_channel = norm_data(r'../bulk-input1')
    label = pd.read_csv(r'../bulk-output/output.csv').values
    label_min = label[:,1:].min(axis=0)
    label_max = label[:,1:].max(axis=0)
    data_info = {'input_min':min_val,
        'input_max':max_val,
        'output_min':label_min,
        'output_max':label_max,
        'num_data':num_data,
        'num_points':num_points,
        'in_channel':in_channel}
    pickle.dump(data_info,open('bulk_datainfo.pkl','wb'))
else:
    data_info = pickle.load(open('bulk_datainfo.pkl','rb'))

input_path = r'../bulk-input1'
output_path = r'../bulk-output/output.csv'
output_path_test = r'../bulk-output/output_test.csv'
point_number = 2600
input_preStr = 'bulk'
data_train = pointdata(input_path,output_path,data_info,point_number,input_preStr)

dl_train = DataLoader(data_train,batch_size = 32,shuffle = True,num_workers=4)


data_valid = pointdata(input_path,output_path_test,data_info,point_number,input_preStr)
dl_valid = DataLoader(data_valid,batch_size = 12,shuffle = False,num_workers=2)

model = pointnet2.get_model(out_channel = 6,in_channel=data_info['in_channel'])
device = torch.device('cuda:0')
# model.load_state_dict(torch.load('train_model_bulk/model_weights_16_0.04825_0.17415.pth'))
# # # model.load_state_dict(torch.load('train_model/model_weights_16_0.434_0.879.pth'))
model.to(device)
his1 = train(model,dl_train,dl_valid,epochs = 10,start=1,lr=8e-5)
his2 = train(model,dl_train,dl_valid,epochs = 50,start=10,lr=1e-5)
his = pd.concat([his1,his2])
his.to_csv("train_log_bulk.txt")
# his2 = train(model,dl_train,dl_valid,epochs = 200,start=101,lr=1e-5)