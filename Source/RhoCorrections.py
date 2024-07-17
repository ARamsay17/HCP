import os
import time

import numpy as np
import xarray as xr

from Source import ZhangRho, PATH_TO_DATA
from Source.ConfigFile import ConfigFile
from Source.Utilities import Utilities
from Source.HDFRoot import HDFRoot


class RhoCorrections:

    @staticmethod
    def M99Corr(windSpeedMean, SZAMean, relAzMean, Propagate = None,
                AOD=None, cloud=None, wTemp=None, sal=None, waveBands=None):
        ''' Mobley 1999 AO'''

        msg = 'Calculating M99 glint correction with complete LUT'
        print(msg)
        Utilities.writeLogFile(msg)

        theta = 40 # viewing zenith angle
        winds = np.arange(0, 14+1, 2)       # 0:2:14
        szas = np.arange(0, 80+1, 10)       # 0:10:80
        phiViews = np.arange(0, 180+1, 15)  # 0:15:180 # phiView is relAz

        # Find the nearest values in the LUT
        wind_idx = Utilities.find_nearest(winds, windSpeedMean)
        wind = winds[wind_idx]
        sza_idx = Utilities.find_nearest(szas, SZAMean)
        sza = szas[sza_idx]
        relAz_idx = Utilities.find_nearest(phiViews, relAzMean)
        relAz = phiViews[relAz_idx]

        # load in the LUT HDF file
        inFilePath = os.path.join(PATH_TO_DATA, 'rhoTable_AO1999.hdf')
        try:
            lut = HDFRoot.readHDF5(inFilePath)
        except:
            msg = "Unable to open M99 LUT."
            Utilities.errorWindow("File Error", msg)
            print(msg)
            Utilities.writeLogFile(msg)

        lutData = lut.groups[0].datasets['LUT'].data
        # convert to a 2D array
        lut = np.array(lutData.tolist())
        # match to the row
        row = lut[(lut[:,0] == wind) & (lut[:,1] == sza) & \
            (lut[:,2] == theta) & (lut[:,4] == relAz)]

        rhoScalar = row[0][5]

        Delta = Propagate.M99_Rho_Uncertainty(mean_vals=[windSpeedMean, SZAMean, relAzMean],
                                              uncertainties=[1, 0.5, 3])/rhoScalar

        # TODO: Model error estimation, requires ancillary data to be switched on. This could create a conflict.
        if not any([AOD is None, wTemp is None, sal is None, waveBands is None]) and \
                ((ConfigFile.settings["bL1bCal"] > 1) or (ConfigFile.settings['SensorType'].lower() == "seabird")):
            # Fix: Truncate input parameters to stay within Zhang ranges:
            # AOD
            AOD = np.min([AOD,0.2])
            # SZA
            if 60 < SZAMean < 70:
                SZAMean = SZAMean
            elif SZAMean >= 70:
                raise ValueError('SZAMean is to high (%s), Zhang correction cannot be performed above SZA=70.')
            # wavelengths in range [350-1000] nm
            newWaveBands = [w for w in waveBands if w >= 350 and w <= 1000]
            # Wavebands clips the ends off of the spectra reducing the amount of values from 200 to 189 for
            # TriOS_NOTRACKER. We need to add these specra back into rhoDelta to prevent broadcasting errors later

            # zhang = RhoCorrections.ZhangCorr(windSpeedMean, AOD, cloud, SZAMean, wTemp, sal,
            zhang, _ = RhoCorrections.ZhangCorr(windSpeedMean, AOD, cloud, SZAMean, wTemp, sal,
                                                relAzMean, newWaveBands)

            import matplotlib.pyplot as plt

            # fig = plt.figure()
            # plt.plot(newWaveBands, rhoScalar * np.ones(len(newWaveBands)), label="M99")
            # plt.plot(newWaveBands, zhang, label="Z17")
            # # plt.title("Mobley99 rho vs Zhang - Tartu")
            # plt.title("Mobley99 rho vs Zhang - HEREON")
            # plt.xlabel("Wavelength (nm)")
            # plt.ylabel("Rho Value")
            # plt.legend()
            # plt.savefig("rho_vals_HEREON.jpg")
            # get the relative difference between mobley and zhang and add in quadrature as uncertainty component

            #  Is |M99 - Z17| the only estimate of glint uncertainty? I thought they had been modeled in MC. -DAA
            #  uncertainties must be in % form (relative and *100) in order to do sum of squares
            pct_diff = (np.abs(rhoScalar - zhang) / rhoScalar)  # relative units
            tot_diff = np.power(np.power(Delta * 100, 2) + np.power(pct_diff * 100, 2), 0.5)
            tot_diff[np.isnan(tot_diff)==True] = 0  # ensure no NaNs are present in the uncertainties.
            tot_diff = (tot_diff/100) * rhoScalar
            # add back in filtered wavelengths
            rhoDelta = []
            i = 0
            for w in waveBands:
                if w >= 350 and w <= 1000:
                    rhoDelta.append(tot_diff[i])
                    i += 1
                else:
                    # in cases where we are outside the range in which Zhang is calculated a placeholder is used
                    rhoDelta.append((np.power(np.power((Delta / rhoScalar) * 100, 2) +
                                     np.power((0.003 / rhoScalar) * 100, 2), 0.5)/100)*rhoScalar)
                    # necessary to convert to relative before propagating, then converted back to absolute
        else:
            # this is temporary. It is possible for users to not select any ancillary data in the config, meaning Zhang
            # Rho will fail. It is far too easy for a user to do this, so I added the following line to make sure the
            # processor doesn't break.
            # 0.003 was chosen because it is the only number with any scientific justification
            # (estimated from Ruddick 2006).

            # Not sure I follow. Ancillary data should be populated regardless of whether an ancillary file is
            # added (i.e., using the model data) - DAA
            rhoDelta = (np.power(np.power((Delta / rhoScalar) * 100, 2)
                        + np.power((0.003 / rhoScalar) * 100, 2), 0.5)/100)*rhoScalar

        # fig = plt.figure()
        # plt.plot(waveBands, rhoDelta)
        # # plt.title("Rho Uncertainty - Tartu")
        # plt.title("Rho Uncertainty - HEREON")
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Rho Uncertainty (absolute)")
        # plt.savefig("rho_unc_HEREON.jpg")
        return rhoScalar, rhoDelta

    @staticmethod
    def threeCCorr(sky750,rhoDefault,windSpeedMean):
        ''' Groetsch et al. 2017 PLACEHOLDER'''
        msg = 'Calculating 3C glint correction'
        print(msg)
        Utilities.writeLogFile(msg)

        if sky750 >= 0.05:
            # Cloudy conditions: no further correction
            if sky750 >= 0.05:
                msg = f'Sky 750 threshold triggered for cloudy sky. Rho set to {rhoDefault}.'
                print(msg)
                Utilities.writeLogFile(msg)
            rhoScalar = rhoDefault
            rhoDelta = 0.003 # Unknown, presumably higher...

        else:
            # Clear sky conditions: correct for wind
            # Set wind speed here
            w = windSpeedMean
            rhoScalar = 0.0256 + 0.00039 * w + 0.000034 * w * w
            rhoDelta = 0.003 # Ruddick 2006 Appendix 2; intended for clear skies as defined here

            msg = f'Rho_sky: {rhoScalar:.6f} Wind: {w:.1f} m/s'
            print(msg)
            Utilities.writeLogFile(msg)

        return rhoScalar, rhoDelta

    @staticmethod
    # def ZhangCorr(windSpeedMean, AOD, cloud, sza, wTemp, sal, relAz, waveBands):
    def ZhangCorr(windSpeedMean, AOD, cloud, sza, wTemp, sal, relAz, waveBands, Propagate = None):
        ''' Requires xarray: http://xarray.pydata.org/en/stable/installing.html
        Recommended installation using Anaconda:
        $ conda install xarray dask netCDF4 bottleneck'''

        msg = 'Calculating Zhang glint correction.'
        print(msg)
        Utilities.writeLogFile(msg)

        # === environmental conditions during experiment ===
        env = {'wind': windSpeedMean, 'od': AOD, 'C': cloud, 'zen_sun': sza, 'wtem': wTemp, 'sal': sal}

        # === The sensor ===
        # the zenith and azimuth angles of light that the sensor will see
        # 0 azimuth angle is where the sun located
        # positive z is upward
        sensor = {'ang': np.array([40, 180 - relAz]), 'wv': np.array(waveBands)}

        # # define uncertainties and create variable list for punpy. Inputs cannot be ordered dictionaries
        varlist = [windSpeedMean, AOD, 0.0, sza, wTemp, sal, relAz, np.array(waveBands)]
        ulist = [1.0, 0.01, 0.0, 0.5, 2, 0.5, 3, None]

        tic = time.process_time()
        rhoVector = ZhangRho.get_sky_sun_rho(env, sensor, round4cache=True)['rho']
        msg = f'Zhang17 Elapsed Time: {time.process_time() - tic:.1f} s'
        print(msg)
        Utilities.writeLogFile(msg)

        # Presumably obsolete (Ashley)? -DAA
        if Propagate is None:
            rhoDelta = 0.003  # Unknown; estimated from Ruddick 2006
        else:
            tic = time.process_time()
            rhoDelta = Propagate.Zhang_Rho_Uncertainty(mean_vals=varlist,
                                                       uncertainties=ulist,
                                                       )
            msg = f'Zhang_Rho_Uncertainty Elapsed Time: {time.process_time() - tic:.1f} s'
            print(msg)
            Utilities.writeLogFile(msg)

        return rhoVector, rhoDelta


if __name__ == '__main__':

    # import Propagate module
    from Source.Uncertainty_Analysis import Propagate
    from scipy.io import savemat

    # initialise punpy
    Prop_Obj = Propagate(M=10, cores=1)

    # set fixed variables
    cloud = None  # not used
    # wTemp = 26.2
    # sal = 35.0
    waveBands = np.array([  # TODO: think if this could cause problems
        351.9, 355.2, 358.5, 361.8, 365.1, 368.4, 371.7, 375.0, 378.3, 381.6, 384.9, 388.2, 391.5, 394.8, 398.1, 401.4,
        404.7, 408.0, 411.3, 414.6, 417.9, 421.2, 424.5, 427.8, 431.1, 434.4, 437.7, 441.0, 444.3, 447.6, 450.9, 454.2,
        457.5, 460.8, 464.1, 467.4, 470.7, 474.0, 477.3, 480.6, 483.9, 487.2, 490.5, 493.8, 497.1, 500.4, 503.7, 507.0,
        510.3, 513.6, 516.9, 520.2, 523.5, 526.8, 530.1, 533.4, 536.7, 540.0, 543.3, 546.6, 549.9, 553.2, 556.5, 559.8,
        563.1, 566.4, 569.7, 573.0, 576.3, 579.6, 582.9, 586.2, 589.5, 592.8, 596.1, 599.4, 602.7, 606.0, 609.3, 612.6,
        615.9, 619.2, 622.5, 625.8, 629.1, 632.4, 635.7, 639.0, 642.3, 645.6, 648.9, 652.2, 655.5, 658.8, 662.1, 665.4,
        668.7, 672.0, 675.3, 678.6, 681.9, 685.2, 688.5, 691.8, 695.1, 698.4, 701.7, 705.0, 708.3, 711.6, 714.9, 718.2,
        721.5, 724.8, 728.1, 731.4, 734.7, 738.0, 741.3, 744.6, 747.9, 751.2, 754.5, 757.8, 761.1, 764.4, 767.7, 771.0,
        774.3, 777.6, 780.9, 784.2, 787.5, 790.8, 794.1, 797.4, 800.7, 804.0, 807.3, 810.6, 813.9, 817.2, 820.5, 823.8,
        827.1, 830.4, 833.7, 837.0, 840.3, 843.6, 846.9, 850.2, 853.5, 856.8, 860.1, 863.4, 866.7, 870.0, 873.3, 876.6,
        879.9, 883.2, 886.5, 889.8, 893.1, 896.4, 899.7, 903.0, 906.3, 909.6, 912.9, 916.2, 919.5, 922.8, 926.1, 929.4,
        932.7, 936.0, 939.3, 942.6, 945.9, 949.2, 952.5, 955.8, 959.1, 962.4, 965.7, 969.0, 972.3, 975.6, 978.9, 982.2,
        985.5, 988.8
    ], dtype=float)

    # create lists to store results
    data_vars = []  # first dimension of N=7 dim tensor, wind speed mean
    unc_vars = []  # first dim uncertainties

    import random as rand
    import xarray as xr
    import os

    # db_path = os.path.join(r"/", "home", "ar17", "PycharmProjects", "FRM4SOC", "HCP", "Z17_LUT.mat")
    # with xr.open_dataset(db_path, engine='netcdf4') as ds:
    #     print(ds)

    windspeed = np.arange(0, 15, 5)  # 4
    AOT = np.array([0, 0.05, 0.1, 0.2, 0.5])  # 5
    SZA = np.arange(0, 65, 5)  # 13
    RELAZ = np.arange(80, 145, 5)  # 13
    SAL = np.arange(0, 45, 5)  # 10
    SST = np.arange(0, 35, 5)  # 8
    data = np.zeros((len(windspeed), len(AOT), len(SZA), len(RELAZ), len(SAL), len(SST), len(waveBands)))
    uncs = np.zeros((len(windspeed), len(AOT), len(SZA), len(RELAZ), len(SAL), len(SST), len(waveBands)))
    for i_windspeed, windSpeedMean in enumerate(windspeed):
        for i_aot, aot in enumerate(AOT):
            for i_sza, sza in enumerate(SZA):
                for i_relaz, relAz in enumerate(RELAZ):
                    for i_sal, sal in enumerate(SAL):
                        for i_wtemp, wtemp in enumerate(SST):
                            # data[i_windspeed, i_aot, i_sza, i_relaz, i_sal, i_wtemp] = [rand.randint(0, 100) for i in range(len(waveBands))]
                            # uncs[i_windspeed, i_aot, i_sza, i_relaz, i_sal, i_wtemp] = [rand.randint(0, 5) for i in range(len(waveBands))]
                            rho, unc = RhoCorrections.ZhangCorr(
                                windSpeedMean,
                                aot,
                                cloud,
                                sza,
                                wtemp,
                                sal,
                                relAz,
                                waveBands,
                                # Propagate=Prop_Obj
                            )
                            data[i_windspeed, i_aot, i_sza, i_relaz, i_sal, i_wtemp] = rho
                            uncs[i_windspeed, i_aot, i_sza, i_relaz, i_sal, i_wtemp] = unc

    da = xr.DataArray(
        data,
        dims=['wind', 'aot', 'sza', 'relAz', 'sal', 'SST', 'wavelength'],
        coords={
            'wind': np.arange(0, 15, 5),
            'aot': [0, 0.005, 0.1, 0.2, 0.5],
            'sza': np.arange(0, 65, 5),
            'relAz': np.arange(80, 145, 5),
            'sal': np.arange(0, 45, 5),
            'SST': np.arange(0, 35, 5),
            'wavelength': waveBands,
        },
        attrs={
            'description': "rho values for given inputs",
            'units': ['ms-1', 'n/a', 'degrees', 'degrees', 'ppm', 'degrees-C']
        },
    )

    du = xr.DataArray(
        uncs,
        dims=['wind', 'aot', 'sza', 'relAz', 'sal', 'SST', 'wavelength'],
        coords={
            'wind': np.arange(0, 15, 5),
            'aot': [0, 0.005, 0.1, 0.2, 0.5],
            'sza': np.arange(0, 65, 5),
            'relAz': np.arange(80, 145, 5),
            'sal': np.arange(0, 45, 5),
            'SST': np.arange(0, 35, 5),
            'wavelength': waveBands,
        },
        attrs={
            'description': "rho uncertainty values for given inputs",
            'units': ['ms-1', 'n/a', 'degrees', 'degrees', 'ppm', 'degrees-C']
        },
    )
    ds = da.to_dataset(name="Glint")
    ds_u = du.to_dataset(name="Uncertainty")

    fp = os.path.join('/home', 'ar17', 'PycharmProjects', 'FRM4SOC')
    print(os.path.join(fp, 'Z17_LUT.nc'))
    ds.to_netcdf(os.path.join(fp, 'Z17_LUT.nc'))  # , 'w', 'NETCDF4')
    ds_u.to_netcdf(os.path.join(fp, 'Z17_UNC_LUT.nc'))  # , 'w', 'NETCDF4')

    da_2 = xr.open_dataset(os.path.join(fp, 'Z17_LUT.nc'))
    du_2 = xr.open_dataset(os.path.join(fp, 'Z17_UNC_LUT.nc'))

    print(da, du)
