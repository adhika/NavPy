"""
Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""
import navpy.gnss.nav as nav
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
    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGNSSNavClass)
    unittest.TextTestRunner(verbosity=2).run(suite)    
