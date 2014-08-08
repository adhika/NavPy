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
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGNSSNavClass)
    unittest.TextTestRunner(verbosity=2).run(suite)    
