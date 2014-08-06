""" GNSS Navigation Module

This module contains GNSS related navigation functions such as calculating
code phase solution and carrier phase ambiguity fixing.

Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""

__authors__ = ['Adhika Lie']
__email__ = ['adhika.lee@gmail.com']
__license__ = 'BSD3'

import numpy as np
import scipy.linalg as la
import navpy as navpy
import navpy.utils as _utils
from ..satorbit import satorbit

def code_phase_LS(raw_meas, gps_ephem, lat=0.0, lon=0.0, alt=0.0, rxclk=0.0):
    """
    Calculate code phase least square solution
    
    Parameters
    ----------
    raw_meas: prn_class object
              Object that contains pseudorange measurements.
    gps_ephem: ephem_class object
               Object that contains the satellite ephemeris.
    
    lat : float, optional
          Initial guess of latitude in degrees
    lon : float, optinal 
          Initial guess of longitude in degrees
    alt : float, optional
          Initial guess of altitude in meters
    rxclk : float, optional
            Initial guess of Receiver clock bias in seconds
    
    Returns
    -------
    lat : float
          Estimated Latitude in degrees
    lon : float
          Estimated Longitude in degrees
    alt : float
          Estimated Altitude in meters
    rxclk : float
            Estimated Receiver Clock Bias in seconds
    """
    # If there are less than 3 satellites, don't do anything
    SV_avbl = np.nonzero(raw_meas.is_dataValid(range(32)))[0]
    
    if(len(SV_avbl) < 3):
        return lat, lon, alt, rxclk
    
    # Begin computing position using Least Squares
    t_tx = raw_meas.get_TOW()*np.ones(len(SV_avbl))
    delta_time = np.zeros(len(SV_avbl))
    
    # Iterate because time at transmission is not known ...
    for k in xrange(5):
        ecef = navpy.lla2ecef(lat,lon,alt)
        t_tx = raw_meas.get_TOW()*np.ones(len(SV_avbl)) - delta_time
        
        # Build satellite information
        clk = satfn.compute_sat_clk_bias(gps_ephem,np.vstack((SV_avbl,t_tx)).T)
        x,y,z = satfn.compute_sat_pos(gps_ephem,np.vstack((SV_avbl,t_tx)).T)
        
        # Build Geometry Matrix H = [LOS 1]
        rho = np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2)  #LOS Magnitude
        H = np.vstack( ( -(x-ecef[0])/rho, -(y-ecef[1])/rho, -(z-ecef[2])/rho, np.ones(len(SV_avbl)) ) ).T
        
        # Calculate estimated pseudorange
        PR_hat = np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2) + rxclk*np.ones(len(SV_avbl))
        
        # Innovation: Difference between the measurement (corrected for satellite clock bias)
        #             and PR_hat
        dy = (raw_meas.get_pseudorange(SV_avbl) + clk*2.99792458e8) - PR_hat
        
        # Measurement Covariance
        RR = raw_meas.get_PR_cov(SV_avbl)

        # Least Square Solution
        dx = la.inv(H.T.dot(RR.dot(H))).dot(H.T.dot(RR)).dot(dy)
        # ... Correction
        ecef += dx[0:3]
        rxclk += dx[3]
        # ... Reclaculate Lat, Lon, Alt
        lat, lon, alt = navpy.ecef2lla(ecef)
        
        # Recalculate delta_time to correct for time at transmission
        x,y,z = compute_sat_pos(gps_ephem,np.vstack((SV_avbl,t_tx)).T)
        delta_time = (np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2))/2.99792458e8
        # ... Let's go to the next iteration ...
     
    return lat, lon, alt, rxclk

def lambda_fix(afloat,Qahat,ncands=2,verbose=False):
    """
    This routine performs an integer ambiguity estimation using the LAMBDA method,
    as developed by the Delft University of Technology, Mathematical Geodesy and Positioning
    
    This version is adaptation of lambda1 function in MATLAB and it is the extended version
    of LAMBDA, intended to be used mainly for research. It has more options than the original
    code.
    
    Parameters
    ----------
    afloat : {(N,)} iterable object
             Float ambiguities.
    Qahat : {(NxN)} ndarray
            Covariance matrix of the ambiguities.
    ncands : int, optional
             Number of candidates searched. Default is 2.
    verbose : bool, optional
              Option for verbose screen print out. Default is False.
        
    Returns
    -------
    afixed : {(N,)}, ndarray
             Estimated integers
    sqnorm : {(N,)} 
             Distance between candidates and float ambiguity input
    Qzhat : {(N,N)} 
             Decorrelated covariance matrix
    Z : {(N,N)}
        Decorrelating transformation matrix
        
    See Also
    --------
    lambda_ldl, lambda_decorrel
    
    Notes
    -----
    [1] Paul de Jonge and Christian Tiberius, ``The LAMBDA method for integer ambiguity 
    estimation,'' Publications of the Delft Geodetic Computing Centre, LGR-Series No. 12, 
    August 1996. 
    URL: http://www.citg.tudelft.nl/over-faculteit/afdelingen/geoscience-and-remote-sensing/research-themes/gps/lambda-method/
    """
    if(verbose):
        print("=========================================================")
        print("LAMBDA - Least Squares Ambiguity Decorrelation Adjustment")
        print("        Based on MATLAB Rel. 2.0b d.d. 19 MAY 1999")
        print("Originally written by:")
        print("           Department of Geodesy and Positioning")
        print("       Faculty of Civil Engineering and Geosciences")
        print("      Delft University of Technology, The Netherlands")
        print("Python Adaptation:")
        print("                       Adhika Lie")
        print("                        July 2014")
        print("=========================================================")
        print("Input Float Ambiguities:")
        print(repr(afloat))
        print("Input Covariance Matrix:")
        print(repr(Qahat))
        print("Number of candidates requested: %d" % ncands)
    
    # Input checks
    afloat,_ = _utils.input_check_Nx1(afloat)
    Qahat = np.array(Qahat)
    if(Qahat.shape[0] != Qahat.shape[1]):
        raise TypeError("Qahat is not square matrix")
    if(len(afloat)!=Qahat.shape[0]):
        raise TypeError("Input dimension not identical")
    if( np.any ((Qahat - Qahat.T)>1e-12)  ):
        raise ValueError("Qahat is not symmetric")
    if( np.sum(la.eig(Qahat)[0]>0) != len(afloat) ):
        raise ValueError("Qahat is not positive definite")
        
    # Make the integer search between +/- 1 by subtracting the whole number
    # out of afloat
    incr = afloat.astype(int)
    afloat = afloat - incr
    if(verbose):
        print("Shifted ambiguities:")
        print(repr(afloat))
        print("Increments:")
        print(repr(incr))
        
    # Compute decorrelation matrix Z
    # Z-transformation based on L-D decomposition of Qahat
    
    afixed = afloat
    return afixed

def lambda_ldl(Qin):
    """
    L'DL decomposition of `Qin`
    
    Parameters
    ----------
    Qin : {(N,N)}, ndarray
          Covariance matrix input
    
    Returns
    -------
    L : {(N,N)}, ndarray
        Lower triangular matrix from decomposition
    D : {(N,N)}, ndarray
        Diagonal matrix from decomposition
    
    Notes
    -----
    The resulting transformation is such that:
    ..math:: Q = L^T D L
    """ 
    Q = Qin.copy()  # This is necessary because we need to modify Q
        
    L = np.zeros(Q.shape)
    D = np.zeros(Q.shape)
    
    for i in range(Q.shape[0]-1,-1,-1):
        
        D[i,i] = Q[i,i]
        
        L[i,0:i+1] = Q[i,0:i+1]/np.sqrt(Q[i,i])
        
        for j in range(0,i):
            Q[j,0:j+1] = Q[j,0:j+1]-L[i,0:j+1]*L[i,j]

        L[i,0:i+1] = L[i,0:i+1]/L[i,i]
        
    if(np.sum(np.diag(D)<1e-10)):
        raise ValueError('Input matrix Q is not positive definite')
        
    return L, D

def lambda_decorrel(Qin):
    """
    Decorrelate covariance matrix of ambiguities and returns the associated 
    matrices.
    
    Parameters
    ----------
    Qin : {(N,N)}, ndarray
          Original covariance matrix of ambiguities
    
    Returns
    -------
    Z : {(N,N), ndarray}
        Transformation matrix
    Q : {(N,N)}, ndarray
        Decorrelated covariance matrix
    
    Notes
    -----
    This routine is based on Fortran routine written by Paul de Jonge (TU Delft)
    and on MATLAB routine written by Kai Borre.
    The implementation is adapted from the MATLAB routine decorrel() written by
    Peter Joosten (TU Delft) in 1999.
    
    The resulting transformation matrix, Z, can be used as follows:
    ..math::
        z = Z^T a
        \hat{z} = Z^T \hat{a}
        Q_{\hat{z}} = Z^T Q_{\hat{a}} Z
    """
    # Get L'DL decomposition of Qin
    L, D = lambda_ldl(Qin)
    
    # Decorrelation procedure
    n = Qin.shape[0]
    Z = np.eye(n)
                
    return Z, Q

def lambda_ztran(Qin,col=None,L=None):
    """
    ZTRAN in Section 3.4 in Ref[1], pp. 13
    
    This algorithm compute a Z-transformation matrix that will make
    the absolute value of all off-diagonal elements of L less than or 
    equal to 0.5
    
    Parameters
    ----------
    Qin : {(N,N)}, ndarray
          Original covariance matrix of ambiguities
    col : {(M,)} iterable object
          Columns that want to be decorrelated, M < N
          Default is all column
    L : {(N,N)}, ndarray
        Input L matrix (L'DL decomposition), for bootstrapping
    
    Returns
    -------
    Z : {(N,N), ndarray}
        Transformation matrix
        
    Notes
    -----
    The Z-transformation matrix is a unit lower triangular matrix (1 on diagonal) where
    the only nonzero off-diagonal term is at (a,b), i.e., 
    :math:`Z(a,b) = \mu`, :math:`a > b`.
    
    Because Z is almost an identity matrix, L*Z is almost exactly L except column b
    which is modified by the value :math:`\mu` in Z.
    ..math::
        LZ(i,b) = L(i,a) Z(a,b) + L(i,b) Z(b,b)
        LZ(i,b) = L(i,a) \mu + L(i,b)
    """
    if(Qin.shape[0]!=Qin.shape[1]):
        raise TypeError("Qin matrix input is not square")
        
    n = Qin.shape[0]
    
    if(L is None):
        L, _ = lambda_ldl(Qin)  # D is never changed
    else:
        if(L.shape[0]!=L.shape[1]):
            raise TypeError("L matrix input is not square")
        if(L.shape[0]!=Qin.shape[1]):
            raise TypeError("L matrix input shape is inconsistent")
    
    Z = np.eye(n) 
    
    if(col is None):
        col = range(0,n)
    else:
        col.sort()
        if(max(col)>=n):
            raise ValueError("Exceed Column")
            
        col = list(set(col))
        
    for b in reversed(col):        # Column
        for a in range(b+1,n):     # Row
            # Decorrelating column b with row a
            mu = -np.round(L[a,b])   # Integer Gauss Approximation
            
            L[a:n,b] += mu*L[a:n,a]  #L[row<a,a] = 0, lower triangular 
            Z[0:n,b] += mu*Z[0:n,a]
            
    return Z