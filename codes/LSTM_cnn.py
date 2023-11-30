# 这个版本改成label是具体数值的


import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.stats.mstats import winsorize
import torchvision
from collections import Counter 
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.model_selection import train_test_split


# Define the finance dataset
class FinanceDataset(Dataset):  #返回的是one_hot,图像，label和label_bool    
    def __init__(self, data,image):
        self.data = torch.Tensor(data)
        self.image = image
        self.mean_0 = None
        self.std_0 = None
        #one-hot的实现应该放在这里，而非getitem,因为getitem是每次取一个样本，和训练过程交织在一起，会很慢
        self.data_x = self.one_hot(self.data[:,:, :5])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):#这个是和train,test在训练中交互的，训练时取得的数据
        return self.data_x[idx,:, :], self.image[idx,:,:,:], self.data[idx,:,5:6], self.data[idx,:, -1:]

    
    def mean(self):
        data_output = self.data[:,:, 5:6]
        non_zero_mask = data_output!= 0
        data_mean = torch.mean(data_output[non_zero_mask].float())
        return data_mean
    
    def std(self):
        data_output = self.data[:,:, 5:6]
        non_zero_mask = data_output!= 0
        data_std = torch.std(data_output[non_zero_mask].float())
        return data_std

    def max(self):
        max_list = []
        for i in range(5):
            max_num = self.data[:,:, i].max()
            max_list.append(max_num)
        return max_list
    
    def min(self):
        min_list = []
        for i in range(5):
            min_num = self.data[:,:, i].min()
            min_list.append(min_num)
        return min_list
    
    def normalize_label(self):
        if self.mean_0 == None:
            mean = self.mean()
        else:
            mean = self.mean_0
            
        if self.std_0 == None:
            std = self.std()
        else:
            std = self.std_0
        
        self.data[:,:,5:6] = (self.data[:,:,5:6]-mean)/std       
    
    def set_mean(self,mean):
        self.mean_0 = mean 
    
    def set_std(self,std):
        self.std_0 = std
    
    
    # 定义一个函数来进行one-hot编码
    def one_hot_encode(self,val):   #val是一个tensor
        # 将值乘以10并取整
        mapped_tensor = (val * 20).floor().long()
        # 对于值为10的元素（原始值为1.0的元素），将其设置为9
        mapped_tensor[mapped_tensor == 20] = 19 
        # 使用F.one_hot()进行one-hot编码
        one_hot_tensor = F.one_hot(mapped_tensor, num_classes=20)
        return one_hot_tensor
    
    
    def one_hot(self,x):
        max_list = self.max()
        min_list = self.min()
        for i in range(0,5):
            x[:,:, i] = (x[:,:, i] - min_list[i])/(max_list[i] - min_list[i])
        one_hot_tensor = self.one_hot_encode(x[:,:, :5])
        reshaped_tensor = one_hot_tensor.view(one_hot_tensor.shape[0],one_hot_tensor.shape[1], -1)    # 使用view
        #print(reshaped_tensor.shape)
        reshaped_tensor = reshaped_tensor.float()
        return reshaped_tensor



# 处理图像数据，将其变成tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])
#然后开始加载图像数据集
train_image_dir="../data_base/train"
test_image_dir="../data_base/test"

# 遍历图像目录并从图像名称中提取(code, date)对
code_date_list_train = []
code_date_list_test = []
train_images = []
test_images = []


for img_name in os.listdir(train_image_dir):
    img_path = os.path.join(train_image_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    train_images.append(img)

    parts = img_name.split('-')
    code = parts[0]
    date = int(parts[1].split('.')[0])  # 去除'.jpg'后缀
    code_date_list_train.append((code, date))
train_images = torch.stack(train_images)   #train_images是一个tensor


for img_name in os.listdir(test_image_dir):
    img_path = os.path.join(test_image_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    test_images.append(img)
    parts = img_name.split('-')
    code = parts[0]
    date = int(parts[1].split('.')[0])  # 去除'.jpg'后缀
    code_date_list_test.append((code, date))
test_images = torch.stack(test_images)   


#准备LSTM的input数据
data = pd.read_csv('data_500.csv',sep=',',encoding='gbk')
data = data[['code','date','open','high','low','close','volume','ret_cf20']]
data['ret_cf20'].fillna(0, inplace=True)   #因为这里空缺值不多，所以直接填充成0，不使用mask了
data['label_bool'] = (data['ret_cf20'] > 0).astype(int)
for column in data.columns[2:7]:
    data[column] = winsorize(data[column], limits=[0.01, 0.01])


windows_train  = [ ]
windows_test = []
pad_test = []
for company, data in data.groupby('code'):
    train_data = data.loc[(data['date'] > 20100101)&(data['date']<= 20121231)]     #(data['date'] > 20100101)&(data['date']<= 20121231)
    window_train = train_data.iloc[:, 2:].values     #window_train中的数据一共有7列
    seq_length = 60
    if (window_train.shape[0] >= 60):
        for i in range(window_train.shape[0] - seq_length + 1):
            small_input = window_train[i:i + seq_length, :]
            date = train_data.iloc[i + seq_length-1, 1]
            code = company   
            if (code,date) in code_date_list_train:
                windows_train.append(small_input) 
    
    test_data = data.loc[(data['date'] > 20130101) & (data['date'] <= 20141231)]   #(data['date'] > 20130101) & (data['date'] <= 20131231)
    window_test = test_data.iloc[:, 2:].values   
    if (window_test.shape[0] > 60):
        for i in range(window_test.shape[0] - seq_length + 1):
            small_input = window_test[i:i + seq_length, :]
            date = test_data.iloc[i + seq_length-1, 1]
            code = company   
            if (code,date) in code_date_list_test:
                windows_test.append(small_input)  

windows_train = np.array(windows_train)
windows_test = np.array(windows_test)

# 将元组列表转换成数组
train_data = np.array(windows_train)
test_data = np.array(windows_test)


# 准备整体数据集
# Shuffle train
idx = np.array(range(train_data.shape[0]))
np.random.shuffle(idx)
train_data = train_data[idx]
train_images = train_images[idx]


# Create the train and test datasets
train_dataset = FinanceDataset(train_data,train_images)
mean = train_dataset.mean()
std = train_dataset.std()
train_dataset.normalize_label()

# 创建一个索引数组
indices = list(range(len(test_data)))
# 使用train_test_split分割索引
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
# 使用这些索引提取对应的数据
val_data = [test_data[i] for i in test_indices]
test_data = [test_data[i] for i in train_indices]
val_data = np.array(val_data)
test_data = np.array(test_data)
val_images = [test_images[i] for i in test_indices]
test_images = [test_images[i] for i in train_indices]    #这里的test_images是一个list
val_images = torch.stack(val_images)
test_images = torch.stack(test_images)  #需要转换成tensor

val_dataset = FinanceDataset(val_data,val_images)
val_dataset.set_mean(mean)
val_dataset.set_std(std)
val_dataset.normalize_label()
    
test_dataset = FinanceDataset(test_data,test_images)
test_dataset.set_mean(mean)
test_dataset.set_std(std)
test_dataset.normalize_label()

# Create the train and test data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers =16)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)


# model
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()

        # CNN layers
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(15,9), stride=(6,3), padding=(20,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(15,9), stride=(6,3), padding=(20,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(15,9), stride=(6,3),padding=(20,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*2*18, 256),
        )
        '''
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5,3), stride=(3,1), padding=(8,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), padding=(8,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1),padding=(8,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5,3), stride=(3,1), padding=(8,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(184320, 256),
        )
        
         # LSTM layers
        self.linear = nn.Linear(input_size, 32)
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Fusion and classification/regression
        self.fc = nn.Sequential(
            nn.Linear(184320 + hidden_size, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, output_size)
        )

    def forward(self, img, seq):
        # Pass image through CNN
        x1 = self.layer1(img)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = x1.view(x1.size(0), -1)  # Flatten tensor for the fully connected layer
        #x1 = x1.view(-1,256*2*18)  
  
        # Pass sequence through LSTM
        x2 = self.linear(seq).view(-1, seq.size(1), 32)
        outputs, (h, c) = self.lstm(x2)
        x2 = outputs[:, -1, :]
        #x2 = self.fc2(outputs[:, -1, :])
        
        # 这里要怎么融合才合适呢？直接对两个模型的结果取均值？还是在更早期的线形层的地方融合？
        # 在线形层融合的方式
        combined = torch.cat((x1, x2), dim=1)
        
        # Final classification/regression layer
        out = self.fc(combined)
        #print(out.shape)
        return out.squeeze()
        
        
# Train the LSTM model
input_size = (windows_train.shape[2] -2) *20   #one-hot是按照10段区分的
hidden_size = 256
output_size = 1   
num_layers = 3
num_epochs = 20
learning_rate = 0.00001   #1e-4
weight_decay = 0.01
#随机抽取300个样本：train+test
#input_size = 32, hidden_size =64: 63.53, val最好是66.99
#input_size = 32, hidden_size =128: 62.46, val最好是66.99


#1层效果没有3层好，3层就够用了，5层太多了
# learning_rate = 0.0001准确率只有50%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss(reduction = 'sum').to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.5, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,betas = [0.9, 0.99])
#scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)


loss_list_train = []
loss_list_val = []
accuracy_list_train = []
accuracy_list_test = []

min_val_loss = 1e9  #保证第一次一定会更新
last_min_ind = -1
early_stopping_epoch = 6   #early stop

best_val_acc = 0
for epoch in range(num_epochs):
    total_loss = 0.0
    training_correct = 0
    val_correct = 0
    num_train = 0
    running_loss = 0.0
    val_current = 0
    num_val = 0
    
    print("Epoch", epoch + 1, "/", num_epochs, ":")
    model.train()
    with tqdm(train_loader) as tepoch:
        for i, (inputs, images,labels,labels_bool) in enumerate(tepoch):
            inputs = inputs.to(device)
            images = images.to(device)
            labels = labels.to(device)
            labels_bool = labels_bool.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            train_outputs = model(images,inputs)
            #print(train_outputs)
            loss = criterion(train_outputs, labels[:,-1,:].squeeze(1))
            total_loss += loss.data
            loss.backward()
            optimizer.step()
            tepoch.set_description("losses {}".format(total_loss / (i + 1)))
            train_outputs = train_outputs*std+ mean
            pred = (train_outputs > 0).float()
            training_correct += torch.sum(pred == labels_bool[:,-1,:].squeeze(1)).item()
            num_train += labels.shape[0]
         #逐步更新lr                    
        scheduler.step()
        
    model.eval()   
    for inputs, images,labels,labels_bool in val_loader:
        inputs = inputs.to(device)
        images = images.to(device)
        labels = labels.to(device)
        labels_bool = labels_bool.to(device)
        val_outputs = model(images,inputs)
        # the class with the highest energy is what we choose as prediction
        loss = criterion(val_outputs,labels[:,-1,:].squeeze(1))
        running_loss += loss.data
        val_outputs = val_outputs*std+ mean
        pred = (val_outputs > 0).float()
        val_correct += torch.sum(pred == labels_bool[:,-1,:].squeeze(1)).item()
        num_val += len(labels)
        
    print("Average training Loss is:", (total_loss / num_train).item(), 
          "Average test Loss is:", (running_loss / num_val).item(), 
        "Train Accuracy is:", (100 * training_correct / num_train), "%",
        "Validation Accuracy is:", (100 * val_correct / num_val), "%")

    if (100 * val_correct / num_val) > best_val_acc:
        best_val_acc = 100 * val_correct / num_val
        best_model = copy.deepcopy(model)
        print(best_val_acc)
        
        
    loss_list_train.append(round((total_loss / num_train).item(),3))
    loss_list_val.append(round((running_loss / num_val).item(),3))
    accuracy_list_train.append(round((training_correct / num_train),2))
    accuracy_list_test.append(round((val_correct / num_val),2))

    final_epoch = 0
    running_loss = running_loss / num_val
    #early stop
    if running_loss < min_val_loss:
        last_min_ind = epoch
        min_val_loss = running_loss    #检测每一次epoch之后val loss有没有变得更小
    elif epoch - last_min_ind >= early_stopping_epoch:
        final_epoch = epoch
        break
    
    
testing_correct = 0
test_loss = 0
best_model.eval()
num_test = 0
batch_test_acc = 0
TP,TN,FP,FN = 0,0,0,0
P,N = 0,0
sum = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():    
    for i, (inputs,images, labels,labels_bool) in enumerate(test_loader):   #最后在测试集上看效果
        inputs = inputs.to(device)
        images = images.to(device)
        labels = labels.to(device)
        labels_bool = labels_bool.to(device)
        # calculate outputs by running images through the network
        outputs = best_model(images,inputs)
        # the class with the highest energy is what we choose as prediction
        outputs = outputs*std+ mean
        pred = (outputs > 0).float()
        testing_correct += torch.sum(pred == labels_bool[:,-1,:].squeeze(1)).item()
        num_test += len(labels_bool)
        TP += ((pred == labels_bool[:,-1,:].squeeze(1))*(labels_bool[:,-1,:].squeeze(1) == 1)).sum()
        TN += ((pred == labels_bool[:,-1,:].squeeze(1))*(labels_bool[:,-1,:].squeeze(1) == 0)).sum()
        FP += ((pred != labels_bool[:,-1,:].squeeze(1))*(labels_bool[:,-1,:].squeeze(1) == 0)).sum()
        FN += ((pred!= labels_bool[:,-1,:].squeeze(1))*(labels_bool[:,-1,:].squeeze(1) == 1)).sum()
        sum += len(pred)
        P += TP+FP
        N += TN+FN
        if i % 100 == 0:
            fp=open('output.log','a+')
            print("{}: The batch accuracy is {}.".format(i, testing_correct/sum), file=fp)
            fp.close()
    fp=open('inference.log','a+')
    print("The test accuracy is {}.\n".format(testing_correct/sum), file=fp)
    print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)), file=fp)
    fp.close()
    print("The test accuracy is {}.\n".format(testing_correct/sum))
    print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)))
    print("Test Accuracy is:", (100 * testing_correct / num_test), "%")
    torch.save(best_model.state_dict(), '_model_2.pth')

#画图
""" Plot loss and accuracy curve """
import matplotlib.pyplot as plt
import numpy as np


if (final_epoch != 0):
    maxEpoch = final_epoch+1
else:
    maxEpoch = epoch +1 

fig = plt.figure()
maxLoss = max(max(loss_list_train),max(loss_list_val))
minLoss = min(min(loss_list_train),min(loss_list_val))
plt.plot(range(1, 1 + maxEpoch), loss_list_train,'-s', label='train loss')
plt.plot(range(1, 1 + maxEpoch), loss_list_val,'-s', label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(range(0, maxEpoch + 1, 2))
plt.axis([0, maxEpoch, minLoss ,maxLoss])
plt.savefig('LSTM_loss_2.png', dpi=300)


fig = plt.figure()
plt.plot(range(1, 1 + maxEpoch), accuracy_list_train, '-s', label='train accuracy')
plt.plot(range(1, 1 + maxEpoch), accuracy_list_test, '-s', label='Valiadtion accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, maxEpoch + 1, 2))
plt.axis([0, maxEpoch, 0.2 ,1])
plt.legend()
plt.savefig('LSTM_acc_2.png', dpi=300)
