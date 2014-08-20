"""
Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""
from navpy import gnss
import navpy.gnss.nav as nav
import navpy.gnss.satorbit as satorbit
import unittest
import numpy as np
import scipy.linalg as la

class TestGNSSNavClass(unittest.TestCase):
    def test_lambda_ldl_LisLowerTriangular(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
        
        L, D = nav.lambda_ldl(Q)
        
        self.assertTrue(~np.any(np.array([L[0,1],L[0,2],L[1,2]])))
        
    def test_lambda_ldl_LDiagonalisOne(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
                      
        L, D = nav.lambda_ldl(Q)
        
        for e1, e2 in zip([1,1,1],np.diag(L)):
            self.assertAlmostEqual(e1,e2,places=10)
    
    def test_lambda_ldl_DDiagonal(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
                      
        L, D = nav.lambda_ldl(Q)
        
        Dout = np.diag(np.diag(D))
        
        np.testing.assert_almost_equal(D,Dout)
    
    def test_lambda_ldl(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
                      
        L, D = nav.lambda_ldl(Q)
        
        Qout = L.T.dot(D).dot(L)
        
        np.testing.assert_almost_equal(Q,Qout)
    
    def test_lambda_ztran(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
                      
        L, D = nav.lambda_ldl(Q)
        
        Z = nav.lambda_ztran(Q, L=L)
        np.testing.assert_array_less(np.array([L[1,0],L[2,0],L[2,1]]),0.5)
        
    def test_lambda_decorrel_checkZ(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
        
        Z, L, D = nav.lambda_decorrel(Q)
        
        Zout = np.array([[-2,  3,  1],
                         [ 3, -3, -1],
                         [-1,  1,  0]])
        np.testing.assert_almost_equal(Z,Zout)
        
    def test_lambda_decorrel_checkDorder(self):
        Q = np.array([[6.290,5.978,0.544],
                      [5.978,6.292,2.340],
                      [0.544,2.340,6.288]])
        
        Z, L, D = nav.lambda_decorrel(Q)
        
        self.assertTrue(D[0,0]>=D[1,1])
        self.assertTrue(D[1,1]>=D[2,2])
    
    def test_code_phase_LS(self):
        ephem_file = '../../satorbit/tests/test_data/brdc1680.13n'
        meas_file = 'test_data/Jun172013_test_AL_HM.txt'
        
        GPSrx = gnss.rx_class()
        
        gps_ephem = satorbit.ephem_class()
        gps_ephem.read_RINEX(ephem_file)
        
        range_data = open(meas_file,'r')
        
        for raw_meas in range_data:
            data = np.fromstring(raw_meas,sep='\t')
    
            # Parse the Data
            for i in xrange(32):
                GPSrx.TOW = data[0]
                GPSrx.rawdata.set_pseudorange(data[6*i+1],data[6*i+4],i)
                GPSrx.rawdata.set_carrierphase(data[6*i+2],data[6*i+5],data[6*i+6],i)
                GPSrx.rawdata.set_doppler(data[6*i+3],i)
        
                GPSrx.rawdata.check_dataValid(i)
        
            SV_avbl = np.nonzero(GPSrx.rawdata.is_dataValid(range(32)))[0]
            
            GPSrx.lat, GPSrx.lon, GPSrx.alt, GPSrx.clkbias = \
                                nav.code_phase_LS(GPSrx,gps_ephem,\
                                                 lat=GPSrx.lat,\
                                                 lon=GPSrx.lon,\
                                                 alt=GPSrx.alt,\
                                                 rxclk=GPSrx.clkbias)

        lat_ref, lon_ref, alt_ref = 44.97989988, -93.22683469, 255.44430556
        
        for e1, e2 in zip([lat_ref, lon_ref, alt_ref],[GPSrx.lat, GPSrx.lon, GPSrx.alt]):
            self.assertAlmostEqual(e1,e2,places=8)
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGNSSNavClass)
    unittest.TextTestRunner(verbosity=2).run(suite)    
