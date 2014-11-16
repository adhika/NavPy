"""
Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""

from navpy import gnss
from navpy.gnss.receiver.novatel import read_gps
import unittest
import numpy as np

class TestNovaTel(unittest.TestCase):
    def test_read_gps(self):
        fid = open('test_data/NovaTel_OEMStar.BIN','rb')
        GPSrx = gnss.rx_class()

        while(1):
            file_pos = fid.tell()
    
            msgID, msgValid = read_gps(fid,GPSrx.rawEpochData)
    
            if(msgValid):
                print("TOW = %f, msgID = %d" % (GPSrx.TOW,msgID))
    
            if( (fid.tell() - file_pos) <= 0 ):
                break

        fid.close()