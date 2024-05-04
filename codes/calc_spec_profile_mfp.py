import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.constants import c as clight
from astropy.constants import e as e_elec
from astropy.constants import h as hplanck
from astropy.cosmology import WMAP9 as cosmo
import glob
import h5py
import yt

clight_kmps=clight.to("km/s").value

#(hplanck*clight/astropy.units.quantity.Quantity(1,"eV")).to("AA")
eV2lambdaAA=lambda eV:12398.42/eV
lambdaAA2eV=lambda AA:12398.42/AA

def drdz(z):
    return clight/cosmo.H(z)/(1+z)

def drdz_matter_dom(z):
    return clight/cosmo.H0/cosmo.Om0**0.5*(1+z)**(-2.5)

def sigma_HI_nu(nu):
    nuHI=13.6
    sigma_HI=6.30e-18*(1.34*(nu/nuHI)**-2.99-0.34*(nu/nuHI)**-3.99) #cm**2
    sigma_HI[nu<nuHI]=0
    return sigma_HI

def calc_tau(nHI,dr_pMpc,eVarray):
    dr=dr_pMpc*3.08e24 #cm
    return nHI*dr*sigma_HI_nu(eVarray) #

def tau_profile(vspec_out,nHI,dr_pMpc):
    lambdaAA_out=eV2lambdaAA(13.6)*(1-vspec_out/2.9979246e5)
    eV_out=lambdaAA2eV(lambdaAA_out)
    return calc_tau(nHI,dr_pMpc,eV_out)

def tau_per_denpixel(vspec_out,pos_pMpc,nHI,vlos,dr_pMpc,Hz):
    vel_space_pos=pos_pMpc*Hz-vlos
    v_relative=vspec_out-vel_space_pos
    lambdaAA_out=eV2lambdaAA(13.6)*(1-v_relative/2.9979246e5)
    eV_out=lambdaAA2eV(lambdaAA_out)
    return calc_tau(nHI,dr_pMpc,eV_out)

def integ_tau(vspec_out,pos_arr_pMpc,nHI,vlos,dr_pMpc,Hz):
    tau_matrix=np.array([tau_per_denpixel(vspec_out,pos_arr_pMpc[i],\
                       nHI[i],vlos[i],dr_pMpc[i],Hz) for i in range(len(pos_arr_pMpc))])
    return np.sum(tau_matrix,axis=0)

vout=np.loadtxt("new_vout.txt")
AAout=np.loadtxt("new_AAout.txt")

aunistr="0.1401"
rootpath="/data/hqchen/mfp_project/data/"
folder="F_a{:s}_50pMpc/".format(aunistr)
zuni=1/float(aunistr)-1
Hz=cosmo.H(zuni).value
print(zuni)

tau_profiles=[]
losList=glob.glob(rootpath+folder+"lightray*h5")

for losName in losList[:]:
    los=yt.load(losName)
    dr=los.r["dl"].to("Mpc").value
    dist=np.cumsum(los.r["dl"].to("Mpc").value)
    nHI=(los.r['gas','RT_HVAR_HI']/yt.units.mp).in_units("cm**-3").v
    vlos=los.r['gas','velocity_los'].to("km/s").v
    cut_inner=0.15
    tau_profiles.append(integ_tau(vout,dist[dist>cut_inner],nHI[dist>cut_inner],vlos[dist>cut_inner],dr[dist>cut_inner],Hz))

np.save("new_tau_profiles_F_a{:s}_noq.npy".format(aunistr),tau_profiles)

qsolos=h5py.File(rootpath+"sp17_1.4e+57_F_a0"+aunistr[2:]+"_xT.hdf5",'r')
qso_t1e7_tau_profiles=[]

for losName in list(qsolos.keys())[:]:
    loslong=yt.load(rootpath+folder+losName+".h5")
    drlong=loslong.r["dl"].to("Mpc").value
    distlong=np.cumsum(loslong.r["dl"].to("Mpc").value)
    nHIlong=(loslong.r['gas','RT_HVAR_HI']/yt.units.mp).in_units("cm**-3").v
    vloslong=loslong.r['gas','velocity_los'].to("km/s").v
    cut_inner=0.15

    
    losinfo=qsolos[losName+'/los.info'][:]
    dist=losinfo[:,0]
    dr=losinfo[:,1]
    xT=qsolos[losName+'/xT_1e7yr'][:]
    nHI_t1e7=losinfo[:,2]*xT[:,1]
    vlos=losinfo[:,-1]
    
    dist_stitch=np.array(list(dist)+list(distlong[distlong>dist[-1]]))
    dr_stitch=np.array(list(dr)+list(drlong[distlong>dist[-1]]))
    nHI_t1e7_stitch=np.array(list(nHI_t1e7)+list(nHIlong[distlong>dist[-1]]))
    vlos_stitch=np.array(list(vlos)+list(vloslong[distlong>dist[-1]]))
    
    qso_t1e7_tau_profiles.append(integ_tau(vout,dist_stitch,nHI_t1e7_stitch,vlos_stitch,dr_stitch,Hz))

np.save("new_tau_profiles_F_a{:s}_qso_Ndot1.4e57_tQ1e7yr.npy".format(aunistr),qso_t1e7_tau_profiles)

#also save tQ=1e5yr
qso_t1e7_tau_profiles=[]

for losName in list(qsolos.keys())[:]:
    loslong=yt.load(rootpath+folder+losName+".h5")
    drlong=loslong.r["dl"].to("Mpc").value
    distlong=np.cumsum(loslong.r["dl"].to("Mpc").value)
    nHIlong=(loslong.r['gas','RT_HVAR_HI']/yt.units.mp).in_units("cm**-3").v
    vloslong=loslong.r['gas','velocity_los'].to("km/s").v
    cut_inner=0.15

    
    losinfo=qsolos[losName+'/los.info'][:]
    dist=losinfo[:,0]
    dr=losinfo[:,1]
    xT=qsolos[losName+'/xT_1e5yr'][:]
    nHI_t1e7=losinfo[:,2]*xT[:,1]
    vlos=losinfo[:,-1]
    
    dist_stitch=np.array(list(dist)+list(distlong[distlong>dist[-1]]))
    dr_stitch=np.array(list(dr)+list(drlong[distlong>dist[-1]]))
    nHI_t1e7_stitch=np.array(list(nHI_t1e7)+list(nHIlong[distlong>dist[-1]]))
    vlos_stitch=np.array(list(vlos)+list(vloslong[distlong>dist[-1]]))
    
    qso_t1e7_tau_profiles.append(integ_tau(vout,dist_stitch,nHI_t1e7_stitch,vlos_stitch,dr_stitch,Hz))

np.save("new_tau_profiles_F_a{:s}_qso_Ndot1.4e57_tQ1e5yr.npy".format(aunistr),qso_t1e7_tau_profiles)
