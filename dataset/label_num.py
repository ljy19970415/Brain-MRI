import numpy as np

label_path='/home/ps/leijiayu/CODE/MedKLIP/Pretrain_MedKLIP/data_file_morecls/label.npy'
labels=np.load(label_path)
total_num=np.zeros((16))
for i in range(labels.shape[0]):
    label=labels[i,:,:]
    num=0
    for j in range(label.shape[1]):
        if 1 in label[:,j]:
            num+=1
    total_num[num-1]+=1
for i in range(len(total_num)):
    print(total_num[i])
        

    