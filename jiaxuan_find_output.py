import glob
import sys
import cv2
import pickle
import numpy as np
from astropy.table import Table, Column

with open(sys.argv[1], 'rb') as ff:
    result = pickle.load(ff)

def find_max(mapp):
    return [np.unravel_index(mapp.argmax(), mapp.shape)[0], np.unravel_index(mapp.argmax(), mapp.shape)[1]]

def gaussian_conv(mapp, radius):
    from astropy.convolution import convolve, Gaussian2DKernel
    cvl = convolve(mapp.astype('float'), Gaussian2DKernel(radius))
    return cvl

def set_nearby_zero(mapp, coor):
    size = mapp.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - coor[0])**2 + (j - coor[1])**2 < 15**2:
                mapp[i][j] = 0
    return mapp


co1 = []
co2 = []
co3 = []
for inx, img in enumerate(result):
    mapp = result[inx][0][:, :, 0].astype(float)
    mapp = gaussian_conv(mapp, 2)
    coor1 = find_max(mapp)
    mapp = set_nearby_zero(mapp, coor1)
    coor2 = find_max(mapp)
    mapp = set_nearby_zero(mapp, coor2)
    coor3 = find_max(mapp)
    co1.append(coor1)
    co2.append(coor2)
    co3.append(coor3)
    if inx % 20 == 0:
        print(inx)
X1 = Column(data=np.array(co1)[:, 1], name='x1')
Y1 = Column(data=np.array(co1)[:, 0], name='y1')
X2 = Column(data=np.array(co2)[:, 1], name='x2')
Y2 = Column(data=np.array(co2)[:, 0], name='y2')
X3 = Column(data=np.array(co3)[:, 1], name='x3')
Y3 = Column(data=np.array(co3)[:, 0], name='y3')
havestar = Column(data=np.ones(len(co1)).astype(int), name='havestar')

test_list = Table.read('./test_list.csv', format='ascii.fast_no_header')
idd = Column(data=test_list['col1'], name='id')

Table([idd, X1, Y1, X2, Y2, X3, Y3, havestar]).write('./output_file_' + sys.argv[2]  +  '.csv', format='csv')
