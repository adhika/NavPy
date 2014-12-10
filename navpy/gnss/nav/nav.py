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

def code_phase_LS(rawEpochData, gps_ephem, lat=0.0, lon=0.0, alt=0.0, rxclk=0.0, SV=None):
    """
    Calculate code phase least square solution
    
    Parameters
    ----------
    rawEpochData: rx_class object
        Contains all the receiver data
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
    SVuse : iterable, float
            PRN used in the computation
    t_tx : iterable, float
           Time at transmission of each SVuse
    """
    # If there are less than 3 satellites, don't do anything
    if(SV is None):
        SVuse = np.nonzero(rawEpochData.is_dataValid(range(32)))[0] 
        print(SVuse)
    else:
        SVuse = SV
    
    if(len(SVuse) < 3):
        return lat, lon, alt, rxclk
    
    # Begin computing position using Least Squares
    delta_time = np.zeros(len(SVuse))
    
    # Iterate because time at transmission is not known ...
    for k in xrange(5):
        ecef = navpy.lla2ecef(lat,lon,alt)
        t_tx = rawEpochData.get_TOW()*np.ones(len(SVuse)) - delta_time
        
        # Build satellite information
        clk = satorbit.compute_sat_clk_bias(gps_ephem,np.vstack((SVuse,t_tx)).T)
        x,y,z = satorbit.compute_sat_pos(gps_ephem,np.vstack((SVuse,t_tx)).T)
        
        # Build Geometry Matrix H = [LOS 1]
        rho = np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2)  #LOS Magnitude
        H = np.vstack( ( -(x-ecef[0])/rho, -(y-ecef[1])/rho, -(z-ecef[2])/rho, np.ones(len(SVuse)) ) ).T
        
        # Calculate estimated pseudorange
        PR_hat = np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2) + rxclk*np.ones(len(SVuse))
        
        # Innovation: Difference between the measurement (corrected for satellite clock bias)
        #             and PR_hat
        dy = (rawEpochData.get_L1CA(SVuse) + clk*2.99792458e8) - PR_hat
        
        # Measurement Covariance
        RR = rawEpochData.get_L1CA_cov(SVuse)

        # Least Square Solution
        dx = la.inv(H.T.dot(RR.dot(H))).dot(H.T.dot(RR)).dot(dy)
        # ... Correction
        ecef += dx[0:3]
        rxclk += dx[3]
        # ... Reclaculate Lat, Lon, Alt
        lat, lon, alt = navpy.ecef2lla(ecef)
        
        # Recalculate delta_time to correct for time at transmission
        x,y,z = satorbit.compute_sat_pos(gps_ephem,np.vstack((SVuse,t_tx)).T)
        delta_time = (np.sqrt((x-ecef[0])**2 + (y-ecef[1])**2 + (z-ecef[2])**2))/2.99792458e8
        # ... Let's go to the next iteration ...
    
    return lat, lon, alt, rxclk, SVuse, t_tx

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
    L : {(N,N)}, ndarray
        Decorrelated unit lower triangular matrix
    D : {(N,N)}, ndarray
        Decorrelated diagonal matrix of variances
    
    Notes
    -----
    The implementation is adapted from the MATLAB routine decorrel() written by
    Peter Joosten (TU Delft) in 1999.
    
    The resulting transformation matrix, Z, can be used as follows:
    ..math::
        z = Z a
        \hat{z} = Z \hat{a}
        Q_{\hat{z}} = Z Q_{\hat{a}} Z^T
    """
    # Decorrelation procedure
    n = Qin.shape[0]
    
    # Get L'DL decomposition of Qin
    L, D = lambda_ldl(Qin)
    # Initialize the Z matrix
    Z = np.eye(n)
    
    isSorted = False     # Flag to indicate that Q is now decorrelated and reordered
    isReordered = False  # Flag to indicate that there is reordering
    
    while (isSorted is False):
        col = n - 1  # Initialize to the last column
        isReordered = False # Initialize this flag
        while (col > 0):
            # ---------------------- DECORRELATION ------------------------
            # Apply ZTRAN to element indicated by `col`
            col -= 1 # ... [Last column need not be decorrelated]
            Z = Z.dot(lambda_ztran(Qin,col=[col],L=L)) 
            
            # ------------------------ REORDERING -------------------------
            # Try swapping `col` with `col+1`
            P = np.eye(n)
            P[col:col+2,col:col+2] = np.array([[0,1],[1,0]]) # Swapping matrix
            
            Qnew = P.dot(L.T.dot(D).dot(L)).dot(P.T)  # This is using decorrelated L
            Lnew, Dnew = lambda_ldl(Qnew)  # Here is the new L'DL
            
            #print("Element %d" % col)
            #print("Reordering matrix P = ")
            #print(P)
            #print("Before reordering, D =")
            #print(D)
            #print("Dnew = ")
            #print(Dnew)
            
            # If the new variance of element `col+1` is less than before swapping...
            # Let's swap them
            if (Dnew[col+1,col+1] < D[col+1,col+1]):
                # This reordering results in descending order of variances
                # print ("Need reordering ...")
                isReordered = True
                Z = Z.dot(P)  # "Add" the swapping matrix P to the transformation Z
                L = Lnew.copy()
                D = Dnew.copy()
                #print("After re-ordering, D = ")
                #print(D)
                
                break; # Break this inner while loop and redo decorrelation from last column
        
        if ((col == 0) and (isReordered is False)):
            isSorted = True
            
    return Z.astype('int'), L, D

def lambda_ztran(Qin,col=None,L=None):
    """
    ZTRAN in Section 3.4 in Ref[1], pp. 13
    
    This algorithm compute a Z-transformation matrix that will make
    the absolute value of all off-diagonal elements of L less than or 
    equal to 0.5
    
    When used in bootstrapped mode, input parameters `col` and `L` 
    must be supplied. Since L is passed as an object,
    L will be modified inside this routine. See example below. 
    
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
        
    Examples
    --------
    >>> import numpy as np
    >>> from navpy.gnss import nav
    >>> import scipy.linalg as la
    
    >>> Q = np.array([[ 5.12617627, -0.62559198, -4.3143666 ,  0.07728635],\
                      [-0.62559198,  2.81791492,  1.91150824,  1.74148283],\
                      [-4.3143666 ,  1.91150824,  9.49349332, -1.86648462],\
                      [ 0.07728635,  1.74148283, -1.86648462,  2.51944188]])
    >>> L, _ = nav.lambda_ldl(Q)                  
    >>> print L
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],\
           [ 2.85849804,  1.        ,  0.        ,  0.        ],\
           [-0.52487319,  0.39474267,  1.        ,  0.        ],\
           [ 0.03067598,  0.6912177 , -0.74083258,  1.        ]])
           
    >>> # Get the Z-matrix in one-go
    >>> Z = nav.lambda_ztran(Q) 
    >>> print Z
    array([[ 1.,  0.,  0.,  0.],\
           [-3.,  1.,  0.,  0.],\
           [ 2.,  0.,  1.,  0.],\
           [ 4., -1.,  1.,  1.]])
    >>> print L.dot(Z)
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],\
           [-0.14150196,  1.        ,  0.        ,  0.        ],\
           [ 0.2908988 ,  0.39474267,  1.        ,  0.        ],\
           [ 0.47535772, -0.3087823 ,  0.25916742,  1.        ]])
    
    >>> # Get the Z-matrix iteratively/bootstrapped
    >>> print L # Still the original one
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],\
           [ 2.85849804,  1.        ,  0.        ,  0.        ],\
           [-0.52487319,  0.39474267,  1.        ,  0.        ],\
           [ 0.03067598,  0.6912177 , -0.74083258,  1.        ]])
           
    >>> Z2 = np.eye(Q.shape[0])
    >>> for i in reversed(range(0,4)):
            Z2 = Z2.dot(nav.lambda_ztran(Q,col=[i],L=L))
    >>> print Z2
    array([[ 1.,  0.,  0.,  0.],\
           [-3.,  1.,  0.,  0.],\
           [ 2.,  0.,  1.,  0.],\
           [ 4., -1.,  1.,  1.]])
    >>> print L # Has been modified inside the lambda_ztran call
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],\
           [-0.14150196,  1.        ,  0.        ,  0.        ],\
           [ 0.2908988 ,  0.39474267,  1.        ,  0.        ],\
           [ 0.47535772, -0.3087823 ,  0.25916742,  1.        ]])
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
        if(max(col)>n):
            raise ValueError("Exceed Column")
            
        col = list(set(col))
        
    for b in reversed(col):        # Column
        for a in range(b+1,n):     # Row
            # Decorrelating column b with row a
            mu = -np.round(L[a,b])   # Integer Gauss Approximation
            
            L[a:n,b] += mu*L[a:n,a]  #L[row<a,a] = 0, lower triangular 
            Z[0:n,b] += mu*Z[0:n,a]
            
    return Z.astype('int')

def lambda_fi71(afloat,L,D,Chi2,ncands=2,verbose=False):
    """
    Algorithm FI71 Section 4.5 of [1]
    
    This algorithm is a direct search from left to right, i.e. from
    the lower bound to the right bound. A set of integers is feasible
    if they lie inside the hyperellipsoid bounded by `Chi2`
    
    To find a full candidate set of integer, the searched is done 
    systematically from the last integer `a[N-1]` to the first `a[0]`.
    Each valid integer will be tried, one at a time, and the adjustment 
    proceeds with the next ambiguity `a[i-1]`, the so-called depth-first
    search. If for a certain ambiguity `a[l]` no valid integers can be
    found, one returns to the previous ambiguity `a[l+1]` and takes the
    next valid integer for this ambiguity. The search terminates when all 
    valid integers encountered, have been treated and one is back at the 
    last ambiguity `a[N-1]`; see how it is done by the BACKTS subroutine 
    below.
    
    The process of validating an integer (i.e. checking if it lies inside
    the hyperellipsoid) is done recursively to speed up the algorithm.
    
    Parameters
    ----------
    afloat : {(N,)} iterable object
             Decorrelated float ambiguities.
    L : {(N,N)}, ndarray
        Decorrelated unit lower triangular matrix
    D : {(N,N)}, ndarray
        Decorrelated diagonal matrix of variances
    Chi2 : int, optional
           Hyperellipsoid volume. Default is 2.
    ncands : Requested number of integer sets
    verbose : bool, optional
    
    Returns
    -------
    afixed : {(N,ncands)}, ndarray
             Integer sets, as many as requested by `ncands`
             Ordered in ascending `sqnorm`
    sqrnom : {(ncands,)}, ndarray         
             The norm-squared of the integer sets. Listed
             in ascending order of magnitude
    """
    # L, D were from Q = L^T * D * L
    # We need for Qinv = Linv * Dinv * Linv^T
    Linv = la.inv(L)
    Dinv = np.diag(1./np.diag(D))
    
    afloat, N = navpy.utils.input_check_Nx1(afloat)
    
    # This is the LHS and RHS of Eq. (4.6), (4.7)
    right = np.zeros(N+1)
    right[-1] = Chi2  # Very last one initialized to Chi2
    
    left = np.zeros(N+1)
    dq = np.hstack((np.diag(Dinv)[1:],1))/np.diag(Dinv)  # d_{i+1}/d_{i}
    
    a_min_ahat = np.zeros(N)
    sum_lji_delta_a = np.zeros(N)
    
    upper_bound = np.zeros(N)
    
    afixed = np.zeros((N,ncands))
    sqnorm = np.nan*np.zeros(ncands)
    
    isEnded = False
    ncan = 0
    
    i = N
    iold = i
    while (isEnded is False):
        i -= 1
        
        if(iold <= i):
            # This is the case, when we happen to backtrack in our search
            # in the BACKTS algorithm below i is incremented
            # So, `i` will become larger than `iold`
            # When this is the case, `a[i+1]` has been incremented by 1
            # So the existing sum_lji_delta_a[i] needs to be incremented by
            # Linv(i+1,i)
            # newSum[i] = Linv_{i+1,i} * ( a[i+1] + 1 -ahat[i+1] ) + sum_from_i+2
            # newSum[i] = Linv_{i+1,i} * 1 + oldSum[i]
            sum_lji_delta_a[i] += Linv[i+1,i]
        else:
            # This is the normal case, when there is no backtracking, you
            # have to calculate the 
            # sum from (j = i+1 to j = N-1) of L_{ji}* ( a[j]-ahat[j] )
            sum_lji_delta_a[i] = 0
            for j in range(i+1,N):
                sum_lji_delta_a[i] += Linv[j,i] * a_min_ahat[j]
        iold = i
        
        right[i] = dq[i] * (right[i+1] - left[i+1]) # cf. Eq (4.9)
        reach = np.sqrt(right[i]) 
        
        # IMPORTANT:
        # The upper and lower bounds are for `a` and NOT `a_min_ahat`
        lower_bound = afloat[i] - reach - sum_lji_delta_a[i] # cf. Eq. (4.14)
        upper_bound[i] = afloat[i] + reach - sum_lji_delta_a[i] 
            # ... Need to save upper bound history in case of backtracking
                    
        a = np.ceil(lower_bound) # First integer in this level after lower bound
        a_min_ahat[i] = a - afloat[i]
        
        if(a > upper_bound[i]):
            # If the first integer exceeds the upper bound
            # there is nothing at this level, so back track 
            # one level up
            # ============ BACKTS Sub-Routine =============
            cand_n = False  # Flag to stop this entire search subroutine (see below)
            c_stop = False  # Flag to stop backtracking
            while ((c_stop is False) and (i < N-1)):
                i += 1  # Get back up one level
                
                # Remember upper bound is for `a` and not `a_min_ahat`
                if( (a_min_ahat[i]+afloat[i]+1) < upper_bound[i] ):
                    # If I `a` becomes `a+1`, will it exceed the upper
                    # bound at this level?
                    # If not, then this `a+1` IS a candidate or a 
                    # feasible integer
                    
                    a_min_ahat[i] += 1  # Make the change of the value of `a` 
                                        # at this level, update the left[i]
                    left[i] = (a_min_ahat[i] + sum_lji_delta_a[i])**2
                    c_stop = True       # We've found a candidate, stop backtracking
                    
                    if(verbose):
                        print("... Exceed upper bound on current level")    
                        print("Backtracked to level i = %d" % i)
                        #print("a = ")
                        #print(a_min_ahat + afloat)
                        #print("\n")
                    
                    if (i is N-1):
                        # Flag indicating that, even though we are at the last level,
                        # we still find a candidate, do not end the iteration just yet
                        cand_n = True
                
            if (i is N-1) and (cand_n is False):
                # We have backtracked all the way to the last level
                # And at this level, we don't have any more candidate.
                # This means, the search has been completed.
                # Now you can set the isEnded flag
                isEnded = True
            # ==============================================
        else:
            # Hey, I haven't reached the upper bound
            # So `a` is a feasible integer
            left[i] = (a_min_ahat[i] + sum_lji_delta_a[i])**2
            if(verbose):    
                print("Searching level i = %d" % i)
                #print("a = ")
                #print(a_min_ahat + afloat)
                #print("\n")
            
        # We have picked one integer in this level (i.e. level i), let's move on
        # to the next level, i.e. level i-1, go to `while(isEnded is False)`
        
        # ... Go back to top, unless you are at i = 0
        
        if (i is 0):
            # Hey, we have reached i = 0, i.e. the lowest level. 
            # 1. Collect ALL integers at this lowest level
            # Calculate the norm ... cf. Eq. (4. 16)
            if(verbose):
                print("... At i = 0, sorting through candidates here")
            t = Chi2 - (right[0] - left[0]) * Dinv[0,0]  # This is using the first integer
                                                       # we found at this level...
            # Remember upper bound is for `a` and not `a_min_ahat`
            while( ( a_min_ahat[0]+afloat[0] ) <= upper_bound[0] ): 
                if(ncan < ncands):
                    afixed[:,ncan] = a_min_ahat + afloat
                    sqnorm[ncan] = t
                    ncan += 1
                else:
                    ipos = np.argmax(sqnorm)
                    if (t < sqnorm[ipos]):
                        afixed[:,ipos] = a_min_ahat + afloat
                        sqnorm[ipos] = t
            
                if(verbose):
                    print("a = "),
                    print(a_min_ahat + afloat), 
                    print(" Norm = %f" % t)
                    
                t += (2*(a_min_ahat[0] + sum_lji_delta_a[0]) + 1 ) *Dinv[0,0] 
                a_min_ahat[0] += 1
            if(verbose):
                print("... Done with the lowest level")
                print("===============================")
                    
            # 2. Back track one, so we are doing a complete systematic sweep
            # ============ BACKTS Sub-Routine =============
            cand_n = False  # Flag to stop this entire search subroutine (see below)
            c_stop = False  # Flag to stop backtracking
            while ((c_stop is False) and (i < N-1)):
                i += 1  # Get back up one level
                
                # Remember upper bound is for `a` and not `a_min_ahat`
                if( (a_min_ahat[i]+afloat[i]+1) < upper_bound[i] ):
                    # If I `a` becomes `a+1`, will it exceed the upper
                    # bound at this level?
                    # If not, then this `a+1` IS a candidate or a 
                    # feasible integer
                    
                    a_min_ahat[i] += 1  # Make the change of the value of `a` 
                                        # at this level, update the left[i]
                    left[i] = (a_min_ahat[i] + sum_lji_delta_a[i])**2
                    c_stop = True       # We've found a candidate, stop backtracking
                    if(verbose):
                        print("Backtracked to level i = %d" % i)
                        #print("a = ")
                        #print(a_min_ahat + afloat)
                    
                    if (i is N-1):
                        # Flag indicating that, even though we are at the last level,
                        # we still find a candidate, do not end the iteration just yet
                        cand_n = True
                
            if (i is N-1) and (cand_n is False):
                # We have backtracked all the way to the last level
                # And at this level, we don't have any more candidate.
                # This means, the search has been completed.
                # Now you can set the isEnded flag
                isEnded = True
            # ==============================================
            
    return afixed[:,sqnorm.argsort()].astype('int'), sqnorm[sqnorm.argsort()]
    
def lambda_chistart(D, L, ain, ncands=2, factor=1.5):
    
    Qinv = la.inv(L.T.dot(D.dot(L)))
    ain, N = navpy.utils.input_check_Nx1(ain)
    
    Chi = []
    
    for k in reversed(range(-1,N)):
        afloat = ain.copy()
        afixed = ain.copy()
        
        #print afloat
        #print afixed
        #print k
        
        for i in reversed(range(0,N)):
            dw = 0
            for j in range(i,N):
                dw += L[j,i] * (afloat[j]-afixed[j])
            #print i
            #print dw
            afloat[i] -= dw
            if(i is not k):
                afixed[i] = np.round(afloat[i])
            else:
                #print "here"
                if ( np.abs(afixed[i]-afloat[i]) < 1e-5 ):
                    afixed[i] = np.round(afixed[i] + 1)
                else:
                    afixed[i] = np.round( afloat[i] + np.sign(afloat[i] - afixed[i]) )
                #print afixed[i]
        
        Chi.append(  (ain-afixed).T.dot(Qinv).dot(ain-afixed) )
    
    Chi = np.array(Chi)
    Chi.sort()    
    return Chi