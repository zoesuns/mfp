import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c as clight
from astropy.constants import e as e_elec
from astropy.constants import h as hplanck
from astropy.cosmology import WMAP9 as cosmo

H0=cosmo.H(0).value
eV2lambdaAA=lambda eV:12398.42/eV
lambdaAA2eV=lambda AA:12398.42/AA

LyC_AA=eV2lambdaAA(13.6)

def kap(drArr,Req,xi,kapbg):
    """
    Calculate kappa profile given inputs.

    Parameters:
    - drArr: array of radial step sizes in pMpc
    - Req: characteristic scale in pMpc, where \Gamma_qso==\Gamma_bkg assuming pure geometric dillusion
    - xi: power-law index
    - kapbg: background opacity in 1/pMpc

    Returns:
    - kapr: array of kappa values at each radial step
    """
    rArr=np.cumsum(drArr)
    kapbg*=3.08e24 #convert to 1/pMpc
    Gq_o_Gbg=np.zeros_like(rArr)
    kapr=np.zeros_like(rArr)
    kapr[0]=0
    Gq_o_Gbg[0]=(rArr[0]/Req)**(-2) #Eq7
    for i in range(1,len(rArr)):
        kapr[i]=kapbg*(1+Gq_o_Gbg[i-1])**(-xi) #Eq4
        Gq_o_Gbg[i]=Gq_o_Gbg[i-1]*(rArr[i]/rArr[i-1])**(-2)*np.exp(-kapr[i-1]*drArr[i-1]) #Eq8
    return kapr#,Gq_o_Gbg

def tau_LyC(wv_rest,zqso,Req,xi,kapbg,xsec_index=2.75,debug=True):
    """
    Calculate LyC optical depth profile using Becker+21 model (which has 3 parameters Req,xi,kapbg).

    Parameters:
    - wv_rest: rest-frame wavelength array where LyC optical depth is calculated
    - zqso: redshift of the quasar
    - Req: [pMpc], model parameter characterizing quasar luminosity, where \Gamma_qso==\Gamma_bkg assuming pure geometric dillusion.
    - xi: dimensionless, power-law index
    - kapbg: [1/pMpc], background opacity 

    Returns:
    - tau_eff: LyC optical depth at each wv_rest
    """
    Hz=cosmo.H(zqso).value
    z912=wv_rest/LyC_AA*(1+zqso)-1
    ngrid=2000
    dzpArr=np.ones(ngrid)*(zqso-z912)/ngrid
    zpArr=z912+(np.cumsum(dzpArr)-dzpArr[0]/2)
    drArr=dzpArr/Hz*clight.to("km/s").value/(1+zqso)
    kap_zp=kap(drArr,Req,xi,kapbg)
    
    integ=(np.sum(kap_zp*(1+zpArr)**(-xsec_index-2.5)*dzpArr))
    tau_eff=clight.to("km/s").value/H0/cosmo.Om0**0.5*(1+z912)**xsec_index*integ
    return tau_eff

def fit_func(wv_rest_arr,zqso,Req,xi,mfp,xsec_index=2.75):
    """
    make sure dAA is small enough
    """
    kapbg=1/mfp # in unit 1/pMpc
    drArr=-(wv_rest_arr[1:]-wv_rest_arr[:-1])/LyC_AA*clight_kmps/Hz
    rArr=np.cumsum(drArr)

    z912=wv_rest_arr/LyC_AA*(1+zqso)-1

    dzpArr=z912[1:]-z912[:-1]

    Gq_o_Gbg=np.zeros_like(rArr)
    kapr=np.zeros_like(rArr)
    kapr[0]=0
    Gq_o_Gbg[0]=(rArr[0]/Req)**(-2) #Eq7
    for i in range(1,len(rArr)):
        kapr[i]=kapbg*(1+Gq_o_Gbg[i-1])**(-xi) #Eq4
        Gq_o_Gbg[i]=Gq_o_Gbg[i-1]*(rArr[i]/rArr[i-1])**(-2)*np.exp(-kapr[i-1]*drArr[i-1]) #Eq8

    kap_zp=kapr
    zpArr=z912[1:]

    integrand=kap_zp*(1+zpArr)**(-xsec_index-2.5)*dzpArr

    integ=-np.cumsum(integrand)

    tau_eff=clight_kmps/H0/cosmo.Om0**0.5*(1+z912[1:])**xsec_index*integ
    
    return np.exp(-tau_eff)

def convert_mfp_kapbg(wv_rest,zqso,kapbg):
    """
    """
    z912=wv_rest/LyC_AA*(1+zqso)-1
    ngrid=2000
    dzpArr=np.ones(ngrid)*(zqso-z912)/ngrid
    zpArr=z912+(np.cumsum(dzpArr)-dzpArr[0]/2)
    drArr=dzpArr/Hz*clight.to("km/s").value/(1+zqso)
    kap_zp=np.ones_like(drArr)*kapbg*3.08e24
    integ=(np.sum(kap_zp*(1+zpArr)**(-5.25)*dzpArr))
    return clight.to("km/s").value/H0/cosmo.Om0**0.5*(1+z912)**2.75*integ

