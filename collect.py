import glob
import cv2
import pickle
import numpy as np
struct = {}
info = np.loadtxt('./list.csv', dtype=str, delimiter=',')
for item in info[1:]:
    if not item[0] in struct:
        struct[item[0]] = []
    struct[item[0]].append((int(item[1]), int(item[2]), item[3]))

blk_list = [
        '0f8deb9380bed80d9c93c28b146f3b71',
        '0d0f07bc631a890372f7a920fad9edd4'
        ]
imgs = {}
def draw(m, x, y, sigma, r):
    src = 255
    for i in range(0, r):
        for alpha in range(0, 256):
            dx = 1.*i*np.cos(alpha/128.*np.pi)
            dy = 1.*i*np.sin(alpha/128.*np.pi)
            dx = int(dx)
            dy = int(dy)
            try:
                m[x+dx,y+dy] = src*np.exp(-1./2*(i**2)/(sigma**2))
            except IndexError:
                continue

for fn in struct:
    if fn in blk_list:
        continue
    img1 = cv2.imread('./merge/%s_b.jpg' % (fn),-1)
    img2 = cv2.imread('./merge/%s_c.jpg' % (fn),-1)
    imgx = np.zeros_like(img1)
    x = struct[fn][0][0]
    try:
        y = struct[fn][0][1]
    except:
        print(struct[fn])
        raise
    draw(imgx, y, x, 2, 8)
    label = struct[fn][0][2]
    rl = 1
    if label in ['noise','ghost','pity']:
        rl = 0
    imgs[fn] = [
            img1, img2, imgx, rl
            ]

with open('merged_data_2.pk','wb') as f:
    pickle.dump(imgs, f)
import IPython
IPython.embed()

