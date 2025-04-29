########################## DEVELOPER USE ONLY ##########################

"""
used to build Z17 look up table for the optimisation of ProcessL2.

"""

# NOTE: I don't know how to do tests for this because anything I write will take weeks to execute - Ashley

########################################################################

# import python packages
import os
# import random as rand
import xarray as xr
import numpy as np

# import RhoCorrections for running of Z17
from Source.ZhangRho import PATH_TO_DATA
from Source.RhoCorrections import RhoCorrections

# import Propagate module for processing uncertainties
from Source.Uncertainty_Analysis import Propagate

if __name__ == '__main__':
    # this is the code used to build the Z17 LUT by repeatedly running the Zhang17 code from HCP. Will run if this file, RhoCorrections.py is __main__
    # i.e. running - python3 HCP/Source/RhoCorrections.py

    # read in current LUT. Figure out it's bounds, then append extra nodes to it!
    # pre = xr.open_dataset(os.path.join(PATH_TO_DATA, "Z17_LUT.nc"), engine='netcdf4')

    # initialise punpy
    # Prop_Obj = Propagate(M=10, cores=1)

    # set fixed variables
    cloud = None  # not used
    waveBands = np.array(
        [350.,  355.,  360.,  365.,  370.,  375.,  380.,  385.,  390.,
         395.,  400.,  405.,  410.,  415.,  420.,  425.,  430.,  435.,
         440.,  445.,  450.,  455.,  460.,  465.,  470.,  475.,  480.,
         485.,  490.,  495.,  500.,  505.,  510.,  515.,  520.,  525.,
         530.,  535.,  540.,  545.,  550.,  555.,  560.,  565.,  570.,
         575.,  580.,  585.,  590.,  595.,  600.,  605.,  610.,  615.,
         620.,  625.,  630.,  635.,  640.,  645.,  650.,  655.,  660.,
         665.,  670.,  675.,  680.,  685.,  690.,  695.,  700.,  705.,
         710.,  715.,  720.,  725.,  730.,  735.,  740.,  745.,  750.,
         755.,  760.,  765.,  770.,  775.,  780.,  785.,  790.,  795.,
         800.,  805.,  810.,  815.,  820.,  825.,  830.,  835.,  840.,
         845.,  850.,  855.,  860.,  865.,  870.,  875.,  880.,  885.,
         890.,  895.,  900.,  905.,  910.,  915.,  920.,  925.,  930.,
         935.,  940.,  945.,  950.,  955.,  960.,  965.,  970.,  975.,
         980.,  985.,  990.,  995., 1000.], dtype=float)  # waveband values already act as nodes for interpolation

    windspeed = np.array([0, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15])  # 10
    AOT = np.array([0, 0.05, 0.1, 0.2, 0.5])  # 5
    SZA = np.arange(10, 65, 5)  # 11  # expanded database would go to 65
    RELAZ = np.arange(80, 145, 5)  # 13
    VZEN = np.array([30, 40]) # 2
    SAL = np.arange(0, 70, 10)  # 7
    SST = np.arange(-40, 60, 20)  # 5
    data = np.zeros((len(windspeed), len(AOT), len(SZA), len(RELAZ), len(SAL), len(SST), len(waveBands)))
    # uncs = np.zeros((len(windspeed), len(AOT), len(SZA), len(RELAZ), len(VZEN), len(SAL), len(SST), len(waveBands)))
    da = {}
    ds = {}

    for i_vzen, vzen in enumerate(VZEN):
        
        for i_windspeed, windSpeedMean in enumerate(windspeed):
            for i_aot, aot in enumerate(AOT):
                for i_sza, sza in enumerate(SZA):
                    for i_relaz, relAz in enumerate(RELAZ):
                        for i_sal, sal in enumerate(SAL):
                            for i_wtemp, wtemp in enumerate(SST):
                                # if (windSpeedMean in pre.wind) and (aot in pre.aot) and (sza in pre.sza) and (relAz in pre.relAz) \
                                # and (vzen in pre.vzen) and (sal in pre.sal) and (wtemp in pre.SST):
                                #     data[i_windspeed, i_aot, i_sza, i_relaz, i_vzen, i_sal, i_wtemp] = pre.Glint.isel(
                                #         wind=i_windspeed, aot=i_aot, sza=i_sza, relAz=i_relaz, vzen=i_vzen, sal=i_sal, SST=i_wtemp
                                #         ).values
                                # else:
                                rho, unc = RhoCorrections.ZhangCorr(
                                    windSpeedMean,
                                    aot,
                                    cloud,
                                    sza,
                                    wtemp,
                                    sal,
                                    relAz,
                                    vzen,
                                    waveBands,
                                    # Propagate=Prop_Obj
                                )
                                data[i_windspeed, i_aot, i_sza, i_relaz, i_sal, i_wtemp] = rho

        da[vzen] = xr.DataArray(
            data,
            dims=['wind', 'aot', 'sza', 'relAz', 'sal', 'SST', 'wavelength'],
            coords={
                'wind': windspeed,
                'aot': AOT,
                'sza': SZA,
                'relAz': RELAZ,
                'sal': SAL,
                'SST': SST,
                'wavelength': waveBands,
            },
            attrs={
                'description': "rho values for given inputs",
                'units': ['ms-1', 'n/a', 'degrees', 'degrees', 'ppm', 'degrees-C']
            },
        )

        ds[vzen] = da[vzen].to_dataset(name="Glint")
        ds[vzen].to_netcdf(os.path.join(PATH_TO_DATA, f'Z17_LUT_{vzen}.nc'))
