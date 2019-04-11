import cv2
import numpy as np
import pickle
import h5py

with open('./merged_data.pk', 'rb') as f:
    d = pickle.load(f)
N = 16*10*1024
data = np.zeros((N, 128, 128, 2), dtype=np.uint8)
label = np.zeros((N, 128, 128, 1), dtype=np.uint8)
n_sample = 16
cnt = 0
for i in d:
    item  = d[i]
    s = item[0].shape
    if np.min(s) < 128:
        continue
    h_range = (64, s[0]-64+1)
    w_range = (64, s[1]-64+1)
    for i in range(0, n_sample):
        h_center = np.random.randint(*h_range)
        w_center = np.random.randint(*w_range)
        x = h_center - 64
        y = w_center - 64
        roi_a = item[0][x:x+128, y:y+128]
        roi_b = item[1][x:x+128, y:y+128]
        roi_l = np.array(item[2][x:x+128, y:y+128])
        if item[3] == 0:
            roi_l[...] = 0
        data[cnt,:,:,0] = roi_a
        data[cnt,:,:,1] = roi_b
        label[cnt,:,:,0] = roi_l
        cnt += 1
print(cnt)
hf = h5py.File('nova_data_v1.h5','w')
hf['data'] = data[:cnt,...]
hf['label'] = label[:cnt,...]
hf.close()

import IPython
IPython.embed()

