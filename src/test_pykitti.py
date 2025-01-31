import pykitti 

basedir = '/home/poseidon/workspace/dataset/KITTI/raw'
date   = "2011_09_26"
drive  = "0018"


dataset = pykitti.raw(basedir, date, drive)

x = list(dataset.velo)
print(x.shape)