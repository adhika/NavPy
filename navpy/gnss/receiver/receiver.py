"""
Copyright (c) 2014 NavPy Developers. All rights reserved.
Use of this source code is governed by a BSD-style license that can be found in
LICENSE.txt
"""

import numpy as np
import navpy.utils as _utils

class rx_class:
    """
    GNSS Receiver Class
    This class contains the GNSS Solution for the receiver
    and the raw measurement object (from prn_class)
    
    Adhika Lie, 06/21/2014
    """
    def __init__(self):
        self.TOW = 0
        self.lat = 0
        self.lon = 0
        self.alt = 0
        self.clkbias = 0
        self.sig_N = 0
        self.sig_E = 0
        self.sig_D = 0
        self.vN = 0
        self.vE = 0
        self.vD = 0
        
        self.rawEpochData = prn_class()
        self.set = []
        self.ttx = []
        
        self.INIT = False
        
class prn_class:
    """
    PRN Class
    This class stores raw measurement data to each GPS Satellite (PRN)
    Fields inside this class are set and accessed via APIs
    Available methods / APIs and their prototypes are:
    1. Pseudorange:
        set_pseudorange(PR,PR_std,sv)
        get_pseudorange(sv)
        get_PR_cov(sv)
    2. Accumulated Delta Range or Carrier Phase:
        set_carrierphase(ADR,ADR_std,sv)
        get_carrierphase(sv)
        get_ADR_cov(sv)
    3. Doppler:
        set_doppler(DR,sv)
        * NEED TO ADD GET *
    4. CNo:
        set_CNo(CNo,sv)
        * NEED TO ADD GET *
    5. Flags (not listed here)
    
    Adhika Lie, 06/21/2014
    """
    def __init__(self):
        self._TOW = np.nan
        self._L1CA = np.nan*np.ones(32)    # L1 C/A code
        self._L1P = np.nan*np.ones(32)     # L1 P code
        self._L1 = np.nan*np.ones(32)      # L1 Phase
        self._L2P = np.nan*np.ones(32)     # L2 P code
        self._L2 = np.nan*np.ones(32)      # L2 Phase 
        self._DR = np.nan*np.ones(32)    # Doppler

        self._L1CA_std = np.nan*np.ones(32)
        self._L1P_std = np.nan*np.ones(32)
        self._L1_std = np.nan*np.ones(32)
        self._L2P_std = np.nan*np.ones(32)
        self._L2_std = np.nan*np.ones(32)

        self._locktime = np.nan*np.ones(32)
        self._CNo = np.nan*np.ones(32)
        self._dataValid = [False for i in range(32)]
        
    ######################################################################
    #                            prn_class APIs
    ######################################################################
    def set_TOW(self,tow):
        self._TOW = tow
        
    def get_TOW(self):
        return self._TOW
        
    # 1. PSEUDORANGE
    # ====================================================================
    # ============================= L1 C/A ===============================
    def set_L1CA(self,PR,PR_std,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        PR,N2 = _utils.input_check_Nx1(PR)
        PR_std,N3 = _utils.input_check_Nx1(PR_std)

        if(np.any(sv>=32)):
            raise TypeError('sv > 32')

        if( (N1!=N2) or (N1!=N3) ):
            raise TypeError('Incompatible size')
            
        if(N1==1):
            sv=[sv]
            PR = [PR]
            PR_std = [PR_std]

        for i in range(N1):
            self._L1CA[sv[i]] = PR[i]
            self._L1CA_std[sv[i]] = PR_std[i]
    
    def get_L1CA(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return np.array([self._L1CA[prn] for prn in sv])
    
    def get_L1CA_cov(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        
        return np.diag([1./self._L1CA_std[prn]**2 for prn in sv])

    # ============================= L1 P ===============================
    def set_L1P(self,PR,PR_std,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        PR,N2 = _utils.input_check_Nx1(PR)
        PR_std,N3 = _utils.input_check_Nx1(PR_std)

        if(np.any(sv>=32)):
            raise TypeError('sv > 32')

        if( (N1!=N2) or (N1!=N3) ):
            raise TypeError('Incompatible size')
            
        if(N1==1):
            sv=[sv]
            PR = [PR]
            PR_std = [PR_std]

        for i in range(N1):
            self._L1P[sv[i]] = PR[i]
            self._L1P_std[sv[i]] = PR_std[i]
    
    def get_L1P(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return np.array([self._L1P[prn] for prn in sv])
    
    def get_L1P_cov(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        
        return np.diag([1./self._L1P_std[prn]**2 for prn in sv])
    
    # ============================= L2 P ===============================
    def set_L2P(self,PR,PR_std,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        PR,N2 = _utils.input_check_Nx1(PR)
        PR_std,N3 = _utils.input_check_Nx1(PR_std)

        if(np.any(sv>=32)):
            raise TypeError('sv > 32')

        if( (N1!=N2) or (N1!=N3) ):
            raise TypeError('Incompatible size')
            
        if(N1==1):
            sv=[sv]
            PR = [PR]
            PR_std = [PR_std]

        for i in range(N1):
            self._L2P[sv[i]] = PR[i]
            self._L2P_std[sv[i]] = PR_std[i]
    
    def get_L2P(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return np.array([self._L2P[prn] for prn in sv])
    
    def get_L2P_cov(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        
        return np.diag([1./self._L2P_std[prn]**2 for prn in sv])
    
    # 2. CARRIER PHASE / ACCUMULATED DOPPLER RANGE 
    # ====================================================================  
    # =============================== L1 =================================
    def set_L1(self,ADR,ADR_std,locktime,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        ADR,N2 = _utils.input_check_Nx1(ADR)
        ADR_std,N3 = _utils.input_check_Nx1(ADR_std)
        locktime,N4 = _utils.input_check_Nx1(locktime)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        if( (N1!=N2) or (N1!=N3) or (N1!=N4)):
            raise TypeError('Incompatible size')
        
        if(N1==1):
            sv=[sv]
            ADR = [ADR]
            ADR_std = [ADR_std]
            locktime = [locktime]
        
        for i in range(N1):
            self._L1[sv[i]] = ADR[i]
            self._L1_std[sv[i]] = ADR_std[i]
            self._locktime[sv[i]] = locktime[i]

    def get_L1(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return np.array([self._L1[prn] for prn in sv])
    
    def get_L1_cov(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        
        return np.diag([1./self._L1_std[prn]**2 for prn in sv])
    
    # =============================== L2 =================================
    def set_L2(self,ADR,ADR_std,locktime,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        ADR,N2 = _utils.input_check_Nx1(ADR)
        ADR_std,N3 = _utils.input_check_Nx1(ADR_std)
        locktime,N4 = _utils.input_check_Nx1(locktime)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        if( (N1!=N2) or (N1!=N3) or (N1!=N4)):
            raise TypeError('Incompatible size')
        
        if(N1==1):
            sv=[sv]
            ADR = [ADR]
            ADR_std = [ADR_std]
            locktime = [locktime]
        
        for i in range(N1):
            self._L2[sv[i]] = ADR[i]
            self._L2_std[sv[i]] = ADR_std[i]
            self._locktime[sv[i]] = locktime[i]

    def get_L2(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return np.array([self._L2[prn] for prn in sv])
    
    def get_L2_cov(self,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        
        return np.diag([1./self._L2_std[prn]**2 for prn in sv])
    
    # 3. DOPPLER      
    # ====================================================================
    def set_doppler(self,DR,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        DR,N2 = _utils.input_check_Nx1(DR)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        if( N1!=N2 ):
            raise TypeError('Incompatible Size')
        
        if(N1==1):
            sv=[sv]
            DR = [DR]
            
        for i in range(N1):
            self._DR[sv[i]] = DR[i]
 
    # 4. CARRIER-TO-NOISE 
    # ====================================================================
    def set_CNo(self,CNo,sv):
        sv,N1 = _utils.input_check_Nx1(sv)
        CNo,N2 = _utils.input_check_Nx1(CNo)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        if( N1!=N2 ):
            raise TypeError('Incompatible Size')
            
        if(N1==1):
            sv=[sv]
            CNo = [CNo]
            
        for i in range(N1):
            self._CNo[sv[i]] = CNo[i]

    # 5. FLAGS           
    # ====================================================================
    def _set_dataValid(self,sv):
        # Is this method ever used??
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')

        if(N1==1):
            sv=[sv]
            
        for prn in sv:
            self._dataValid[prn] = True
    
    def _reset_dataValid(self,sv):
        # Is this method ever used??
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        for prn in sv:
            self._dataValid[prn] = False
    
    def check_dataValid(self,sv):
        # Consider renaming this method to "set_dataValidity"
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]    
            
        for prn in sv:
            self._dataValid[prn] = ~( np.any(np.isnan(self._L1CA[prn])) or np.any(np.isnan(self._L1[prn])) ) 
        
    def is_dataValid(self,sv):
        # Consider renaming this method to "get_dataValidity"
        sv,N1 = _utils.input_check_Nx1(sv)
        if(np.any(sv>=32)):
            raise TypeError('sv > 32')
        
        if(N1==1):
            sv=[sv]
        return [self._dataValid[prn] for prn in sv]