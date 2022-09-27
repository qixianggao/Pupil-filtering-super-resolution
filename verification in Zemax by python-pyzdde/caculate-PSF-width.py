import importlib
import io

import numpy as np
from pprint import pprint

import pyzdde.zdde as pyz

import sys

from matplotlib import pyplot as plt

importlib.reload(sys)
dd = sys.getdefaultencoding()

ln = pyz.createLink()
ln.apr = True
ln.zLoadFile('perfect-optical-system.zmx')
cc = ln.zModifyFFTPSFCrossSecSettings(settingsFile='1.cfg', dtype=0, sample=5)
print(cc)
# m = 14.2  # 20.9 14.9 15.6 14.2
# ln.zSetSurfaceParameter(surfNum=1, param=20, value=m)
# ln.zSetSurfaceParameter(surfNum=1, param=23, value=m)
#
#
#
# ln.zSetHuygensPSFSettings(field=1)
# t = ln.zGetPSF(which='huygens', settingsFile=None, txtFile=None,keepFile=False, timeout=120)
# # np.savetxt(r'HPSF.txt', t[1])
# # pprint(t[1])
# a = np.max(t[1])
# print(a)

# cc=ln.zModifyFFTPSFCrossSecSettings(settingsFile='l.cfg', dtype=0,row=None,
#                                    sample=None, wave=None, field=None, pol=None,
#                                    norm=0, scale=None)
# pprint(cc)
# a = ln.zGetPSFCrossSec(which='fft', settingsFile='l.cfg', txtFile=None,keepFile=False, timeout=120)
a = ln.zGetPSFCrossSec(which='fft', settingsFile='1.cfg', txtFile=None, keepFile=False, timeout=120)
pprint(a[2][109])
# pprint(a[2])
strl = np.max(a[2])
print(strl)
loc = np.argmax(a[2])
print(loc)
ii = 1
while a[2][loc + ii] < a[2][loc + ii - 1]:
    ii = ii + 1
locright = loc + ii - 1
# print(locright)
ii = 1
while a[2][loc - ii] < a[2][loc - ii + 1]:
    ii = ii + 1
locleft = loc - ii + 1
# print(locleft)
Glong = a[1][locright] - a[1][locleft]
print(Glong)
# t = ln.zGetPSF(which='huygens', settingsFile=None, txtFile=None,keepFile=False, timeout=120)
# pprint(t)

iii = 1
while a[2][loc + iii] > 0.2*1:
    iii = iii + 1
locr = loc + iii - 1
# print(locright)
iii = 1
while a[2][loc - iii] > 0.2*1:
    iii = iii + 1
locl = loc - iii + 1
# print(locleft)
Glong = a[1][locr] - a[1][locl]
print(Glong)
