import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import yt
import glob
import matplotlib.pyplot as plt
from scipy import interpolate

def parse_ionDens(numberDens,ionicFrac):
    # input numberDens=[nH, nHe]
    # input ionicFrac=[xHI, xHeI, xHeII]
    # return nHI,nHII,nHeI,nHeII,nHeIII,ne,ntot
    nHI=numberDens[0]*ionicFrac[0]
    nHII=numberDens[0]*(1-ionicFrac[0])
    nHeI=numberDens[1]*ionicFrac[1]
    nHeII=numberDens[1]*ionicFrac[2]
    nHeIII=numberDens[1]*(1-ionicFrac[1]-ionicFrac[2])
    ne=nHII+nHeII+2*nHeIII
    ntot=numberDens[0]+numberDens[1]+ne
    return nHI,nHII,nHeI,nHeII,nHeIII,ne,ntot


def sigma_HI_nu(nu):
    # atomic data: photo-ionization xsec of HI 
    # input nu in eV (> 13.6 eV!!! only works when nu>nuHI)
    # output sigma_HI in cm^2
    nuHI=13.6
    sigma_HI=6.30e-18*(1.34*(nu/nuHI)**-2.99-0.34*(nu/nuHI)**-3.99) #cm**2
    return sigma_HI

def sigma_HeI_nu(nu):
    # input nu in eV (> 24.6 eV!!!)
    nuHeI=24.6
    sigma_HI=7.03e-18*(1.66*(nu/nuHeI)**-2.05-0.66*(nu/nuHeI)**-3.05) #cm**2
    return sigma_HI

def sigma_HeII_nu(nu):
    # input nu in eV (> 54.4 eV!!!)
    nuHeII=54.4
    sigma_HI=1.50e-18*(1.34*(nu/nuHeII)**-2.99-0.34*(nu/nuHeII)**-3.99) #cm**2
    return sigma_HI

def ion_potential(species):
    ionpdict={'HI':13.6,'HeI':24.6,'HeII':54.4}
    return ionpdict[species]

def sigma_nu(nu,species):
    sigmadict={'HI':sigma_HI_nu,'HeI':sigma_HeI_nu,'HeII':sigma_HeII_nu}
    return sigmadict[species](nu)
    
def eGamma_HI(T):
    T5=T/1e5
    eGamma=1.17e-10*T**0.5*np.exp(-157809.1/T)/(1+T5**0.5)
    return eGamma

def eGamma_HeI(T):
    T5=T/1e5
    eGamma=4.76e-11*T**0.5*np.exp(-285335.4/T)/(1+T5**0.5)
    return eGamma

def eGamma_HeII(T):
    T5=T/1e5
    eGamma=1.14e-11*T**0.5*np.exp(-631515.0/T)/(1+T5**0.5)
    return eGamma

def colli_ioniz_rate(T,species):
    eGammaDict={'HI':eGamma_HI,'HeI':eGamma_HeI,'HeII':eGamma_HeII}
    return eGammaDict[species](T)

def alphaA_HI_Abel97(T):
    ### fit doesn't work for T>1e6K
    T_eV=T*8.61733e-5
    recHI=np.exp(-28.6130338 - 0.72411256*np.log(T_eV) - 2.02604473e-2*np.log(T_eV)**2
        - 2.38086188e-3*np.log(T_eV)**3 - 3.21260521e-4*np.log(T_eV)**4 - 1.42150291e-5*np.log(T_eV)**5
        + 4.98910892e-6*np.log(T_eV)**6 + 5.75561414e-7*np.log(T_eV)**7 - 1.85676704e-8*np.log(T_eV)**8
        - 3.07113524e-9*np.log(T_eV)**9)
    return recHI

def alpha_r_HeI_Abel97(T):
    ### radiative recombination, 10% error within 1e3-1e5 K, terrible outside
    T_eV=T*8.61733e-5
    return (3.925e-13*T_eV**-0.6353)
   
def alpha_d_HeI_Abel97(T):
    ### dielectronic recombination
    ### !!! more than a factor of 2 different above 1e5 K
    T_eV=T*8.61733e-5
    drecHeI=1.544e-9*T_eV**-1.5*np.exp(-48.596/T_eV)*(0.3+np.exp(8.1/T_eV))
    drecHeI[T<1e3]=0;
    return drecHeI

def alpha_HeI_Abel97(T):
    return (alpha_r_HeI_Abel97(T)+alpha_d_HeI_Abel97(T))

def alpha_HeII_Abel97(T):
    return (2*alphaA_HI_Abel97(T/4))

def recomb_rate(T,species):
    alphadict={'HI':alphaA_HI_Abel97,'HeI':alpha_HeI_Abel97,'HeII':alpha_HeII_Abel97, \
                "HeI_r":alpha_r_HeI_Abel97, "HeI_d":alpha_d_HeI_Abel97}
    return alphadict[species](T)

def init_photo_bkg(los):
    T=los["T"]
    nH=los["nH"]
    nHe=los["nHe"]
    xHI=los["xHI"]
    xHeI=los["xHeI"]
    xHeII=los["xHeII"]
    nHI,nHII,nHeI,nHeII,nHeIII,ne,ntot=parse_ionDens([nH,nHe],[xHI,xHeI,xHeII,T]) 
    Gamma_HI=recomb_rate(T,"HI")*(1-xHI)*ne/xHI-ne*eGamma_HI(T)
    Gamma_HeI=recomb_rate(T,"HeI")*(xHeII)*ne/xHeI-ne*eGamma_HeI(T)
    Gamma_HeII=recomb_rate(T,"HeII")*(1-xHeI-xHeII)*ne/xHeII-ne*eGamma_HeII(T)

    Gamma_HI[Gamma_HI<0]=0
    Gamma_HeI[Gamma_HeI<0]=0
    Gamma_HeII[Gamma_HeII<0]=0
    Heating_bkg=np.array([np.zeros_like(T),np.zeros_like(T),np.zeros_like(T)])
    return np.array([Gamma_HI,Gamma_HeI,Gamma_HeII]),Heating_bkg


rootpath="/data/hqchen/mfp_project/data/"
folderL=glob.glob(rootpath+"F_a*_50pMpc/")

sample_dist=np.linspace(1,49,4801)

for folderp in folderL:
    #folder="F_a0.1401_50pMpc/"
    folder=folderp.split(rootpath)[-1]
    auni=float(folder.split("_50pMpc/")[0].split("F_a")[-1])
    zuni=1/auni-1
    Hz=cosmo.H(zuni).value

    gammalist=[]
    losList=glob.glob(rootpath+folder+"lightray*h5")
    for losName in losList[:]:
        lr=yt.load(losName)
        los={}
        los["dr"]=lr.r['dl'].in_units("Mpc").v
        los["dist"]=np.cumsum(los["dr"])-los["dr"][0]/2.
        los["T"]=lr.r['temperature'].in_units("K").v
        los["nH"]=((lr.r['gas','RT_HVAR_HI']+lr.r['gas','RT_HVAR_HII'])/yt.units.mp).in_units("cm**-3").v
        los["nHe"]=((lr.r['gas','RT_HVAR_HeI']+lr.r['gas','RT_HVAR_HeII']+lr.r['gas','RT_HVAR_HeIII'])/yt.units.mp/4).in_units("cm**-3").v
        los["xHI"]=lr.r['gas','RT_HVAR_HI']/(lr.r['gas','RT_HVAR_HI']+lr.r['gas','RT_HVAR_HII'])
        los["xHeI"]=lr.r['gas','RT_HVAR_HeI']/(lr.r['gas','RT_HVAR_HeI']+lr.r['gas','RT_HVAR_HeII']+lr.r['gas','RT_HVAR_HeIII'])
        los["xHeII"]=lr.r['gas','RT_HVAR_HeII']/(lr.r['gas','RT_HVAR_HeI']+lr.r['gas','RT_HVAR_HeII']+lr.r['gas','RT_HVAR_HeIII'])
        gamma=init_photo_bkg(los)[0][0]
        dist=np.cumsum(lr.r['dl'].in_units("Mpc").v)
        func=interpolate.interp1d(dist,gamma)
        gammalist+=list(func(sample_dist))
    
    
    np.save("unisample_gamma_bkg_z={:3.1f}.npy".format(zuni),gammalist)
