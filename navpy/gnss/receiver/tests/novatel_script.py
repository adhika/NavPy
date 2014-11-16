import navpy
from navpy import gnss
from navpy.gnss.receiver.novatel import read_gps
import navpy.gnss.nav as gnssnav
import navpy.gnss.satorbit as satorbit
import numpy as np
import matplotlib.pyplot as plt
import pylab 

plt.close('all')

ephem_file = '../../satorbit/tests/test_data/brdc1680.13n'
gps_ephem = satorbit.ephem_class()
gps_ephem.read_RINEX(ephem_file)

fid = open('test_data/NovaTel_OEMStar.BIN','rb')
GPSrx = gnss.rx_class()

#fig, ax = plt.subplots(2,5)
#fig.canvas.setFixedSize(1440,720)
#for i in xrange(0,2):
#    for j in xrange(0,5):
#        ax[i,j].set_aspect('equal')
#        ax[i,j].set_title('PRN')
#fig.show()
#fig.tight_layout()
#fig.canvas.draw()

t_sv_azel = {'TOW':[]}
for i in xrange(0,32):
    t_sv_azel[i] = []
    
LAT_REF = 44.979905682483007
LON_REF = -93.226819258089307
ALT_REF = 300.0
posfig, posax = plt.subplots(1,1)
plt.show()

while(1):
    file_pos = fid.tell()

    msgID, msgValid = read_gps(fid,GPSrx)

    if ((msgValid is False)):
        print("CRC Error")
        continue
    
    if(msgID is not 43):
        continue
        
    SV_avbl = np.nonzero(GPSrx.rawdata.is_dataValid(range(32)))[0]
    if(len(SV_avbl)<3):
        #print("TOW = %11.3f, Less than 3 satellites" % GPSrx.TOW)
        continue
    sv_t = np.vstack((SV_avbl,GPSrx.TOW*np.ones(len(SV_avbl)))).T # Often used

    # Azimuth - ELevation calculation
    # Supply ephemeris, so the first three arguments don't matter
    azel = satorbit.calc_azel(0,0,0,\
                                GPSrx.lat,GPSrx.lon,GPSrx.alt,\
                                ephem=gps_ephem,\
                                sv_t=sv_t)
    data = np.vstack((GPSrx.TOW*np.ones(len(SV_avbl)),SV_avbl,np.rad2deg(azel.T))).T
    #t_sv_azel = np.vstack((t_sv_azel,data))
    #print("TOW = %f, msgID = %d" % (GPSrx.TOW,msgID))
    
    # =========================== NAVIGATION ALGORITHM ===========================
    # Position Calculation
    GPSrx.lat, GPSrx.lon, GPSrx.alt, GPSrx.clkbias = \
                        gnssnav.code_phase_LS(GPSrx,gps_ephem,\
                                         lat=GPSrx.lat,\
                                         lon=GPSrx.lon,\
                                         alt=GPSrx.alt,\
                                         rxclk=GPSrx.clkbias)
                                         
    if((GPSrx.TOW-int(GPSrx.TOW)) < 1e-5):
        print("TOW = %11.3f, Lat: %13.8f deg, Lon: %13.8f deg, Alt: %7.2f" % \
                    (GPSrx.TOW,GPSrx.lat,GPSrx.lon,GPSrx.alt))
        
        NED =  navpy.lla2ned(GPSrx.lat,GPSrx.lon,GPSrx.alt,LAT_REF,LON_REF,ALT_REF)
        plt.scatter(NED[1],NED[0],figure=posfig)
        pylab.draw()  # !!!!!!!!!!!!  FIGURE DOESN'T UPDATE  !!!!!!!!! 
        
    # Check for EOF
    if( (fid.tell() - file_pos) <= 0 ):
        break

fid.close()