"""
Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""

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
    raw_meas: prn_class object, containing the pseudorange measurements
    gps_ephem: ephem_class object, containing the satellite ephemeris
    
    lat : (optional, degrees), initial guess of latitude
    lon : (optional, degrees), initial guess of longitude
    alt : (optional, m), initial guess of altitude
    rxclk : (optional, seconds), initial guess of Receiver clock bias
    
    Returns
    -------
    lat : (deg) Latitude
    lon : (deg) Longitude
    alt : (m) Altitude
    rxclk : (sec) Receiver Clock Bias
    
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
    afloat : { (N,) } Float ambiguities
    Qahat : { (NxN) }, Covariance matrix of the ambiguities
    ncands : optional
        Number of candidates searched. Default is 2
    verbose : optional
        True or False
        
    Returns
    -------
    afixed : { (N,) } Estimated integers
    sqnorm : { (N,) } Distance between candidates and float ambiguity input
    Qzhat : { (N,N) } Decorrelated covariance matrix
    Z : { (N,N) } Decorrelating transformation matrix
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
    LDL decomposition such that
    Qin = L'*D*L
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