import importlib
import numpy as np
from pprint import pprint
import pyzdde.zdde as pyz
import sys
import scipy.io
import importlib
importlib.reload(sys)
dd = sys.getdefaultencoding()
# -*- coding: utf-8 -*-
ln = pyz.createLink()
ln.apr = True
ln.zLoadFile('perfect_system.zmx')
# cc = ln.zModifyFFTPSFSettings(settingsFile='1.cfg', dtype=None, sample=None,
#                               wave=None, field=None, surf=None, pol=None,
#                               norm=None, imgDelta=None)
# print(cc)
for w in range(3):
    w = w + 1
    for fd in range(3):
        ln.zSetSurfaceParameter(surfNum=2, param=25, value=0.1+fd*0.03)
        fd = fd + 1

        ln.zSetFFTPSFSettings(settingsFile='2.cfg', dtype=0, sample=3,
                              wave=w, field=1, surf=0, pol=0,
                              norm=0, imgDelta=None)
        # print(a)

        b = ln.zGetPSF(which='fft', settingsFile='2.cfg', txtFile=None,
                       keepFile=False, timeout=120)
        # print(b[0])
        with open(r'C:\\Users\Administrator\\Nutstore\\1\\我的坚果云\\像差模拟\\球差\\test.txt', 'a', encoding='ANSI') as fl:
            fl.write(str(b[0]) + '\n')
        # with open(r'PSF.txt', 'w', encoding='ANSI') as f:
        #     f.write(b[1])

        scipy.io.savemat('C:\\Users\Administrator\\Nutstore\\1\\我的坚果云\\像差模拟\\球差\\'+'PSFw' + str(w) +'f' + str(fd) + '.mat', mdict={'PSFw'+ str(w) +'f' + str(fd): b[1]})
