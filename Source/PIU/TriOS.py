# python packages
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

# maths
import numpy as np

# typing
from typing import Optional, Union, Any

# Source files
from Source.Utilities import Utilities
from Source.HDFGroup import HDFGroup

# PIU files
from Source.PIU.BaseInstrument import BaseInstrument


class TriOS(BaseInstrument):

    def __init__(self):
        super().__init__()
        self.name = "TriOS"

    def lightDarkStats(self, grp: HDFGroup, XSlice: OrderedDict, sensortype: str) -> dict[str: Union[np.array, dict]]:
        """
        TriOS specific method to get statistics from ensemble

        :param grp: HDFGroup of data for the entire RAW file
        :param XSlice: OrderedDict of data for the ensemble
        :param sensortype: name of sensortype, i.e. ES, LI or LT
        """

        (
            raw_cal, 
            raw_back,
            raw_data,
            raw_wvl,
            int_time,
            int_time_t0,
            DarkPixelStart,
            DarkPixelStop,
        ) = TriOSUtils.readParams(grp, XSlice, sensortype)

        del grp, XSlice  # delete unused data to save memory

        # check size of data
        nband = len(raw_back)  # indexes changed for raw_back as is brought to L2
        nmes = len(raw_data)
        if nband != len(raw_data[0]):
            print("ERROR: different number of pixels between dat and back")
            return None

        # Data conversion
        mesure = raw_data/65535.0
        calibrated_mesure = np.zeros((nmes, nband))
        calibrated_light_measure = np.zeros((nmes, nband))
        back_mesure = np.zeros((nmes, nband))

        for n in range(nmes):
            # Background correction : B0 and B1 read from "back data"
            back_mesure[n, :] = raw_back[:, 0] + raw_back[:, 1]*(int_time[n]/int_time_t0)
            back_corrected_mesure = mesure[n] - back_mesure[n, :]

            # Offset substraction : dark index read from attribute
            offset = np.mean(back_corrected_mesure[DarkPixelStart:DarkPixelStop])
            offset_corrected_mesure = back_corrected_mesure - offset

            # Normalization for integration time
            normalized_mesure = offset_corrected_mesure*int_time_t0/int_time[n]
            normalised_light_measure = back_corrected_mesure*int_time_t0/int_time[n]  # do not do the dark substitution as we need light data

            # Sensitivity calibration
            calibrated_mesure[n, :] = normalized_mesure/raw_cal  # uncommented /raw_cal L1985-6
            calibrated_light_measure[n, :] = normalised_light_measure/raw_cal

        # get light and dark data before correction
        light_avg = np.mean(calibrated_light_measure, axis=0)  # [ind_nocal == False]
        if nmes > 25:
            light_std = np.std(calibrated_light_measure, axis=0) / pow(nmes, 0.5)  # [ind_nocal == False]
        elif nmes > 3:
            light_std = np.sqrt(((nmes-1)/(nmes-3))*(np.std(calibrated_light_measure, axis=0) / np.sqrt(nmes))**2)
        else:
            msg = "too few scans to make meaningful statistics"
            print(msg)
            Utilities.writeLogFile(msg)
            return False
        # ensure all TriOS outputs are length 255 to match SeaBird HyperOCR stats output
        ones = np.ones(nband)  # to provide array of 1s with the correct shape
        dark_avg = ones * offset
        if nmes > 25:
            dark_std = ones * np.std(back_corrected_mesure[DarkPixelStart:DarkPixelStop], axis=0) / pow(nmes, 0.5)
        else:  # already checked for light data so we know nmes > 3
            dark_std = np.sqrt(((nmes-1)/(nmes-3))*(
                    ones * np.std(back_corrected_mesure[DarkPixelStart:DarkPixelStop], axis=0)/np.sqrt(nmes))**2)
        # adjusting the dark_ave and dark_std shapes will remove sensor specific behaviour in Default and Factory

        std_signal = {}
        for i, wvl in enumerate(raw_wvl):
            std_signal[wvl] = pow(
                (pow(light_std[i], 2) + pow(dark_std[i], 2)), 0.5) / np.average(calibrated_mesure, axis=0)[i]

        return dict(
            ave_Light=np.array(light_avg),
            ave_Dark=np.array(dark_avg),
            std_Light=np.array(light_std),
            std_Dark=np.array(dark_std),
            std_Signal=std_signal,
        )


class TriOSUtils:
    def __init__(self):
        pass

    @staticmethod
    def readParams(grp, data, s):
        raw_cal = grp.getDataset(f"CAL_{s}").data
        raw_back = np.asarray(grp.getDataset(f"BACK_{s}").data.tolist())
        raw_data = np.asarray(list(data['data'].values())).transpose()  # data is transpose of old version

        raw_wvl = np.array(pd.DataFrame(grp.getDataset(s).data).columns)
        int_time = np.asarray(grp.getDataset("INTTIME").data.tolist())
        DarkPixelStart = int(grp.attributes["DarkPixelStart"])
        DarkPixelStop = int(grp.attributes["DarkPixelStop"])
        int_time_t0 = int(grp.getDataset(f"BACK_{s}").attributes["IntegrationTime"])

        # sensitivity factor : if raw_cal==0 (or NaN), no calibration is performed and data is affected to 0
        ind_zero = np.array([rc[0] == 0 for rc in raw_cal])  # changed due to raw_cal now being a np array
        ind_nan = np.array([np.isnan(rc[0]) for rc in raw_cal])
        ind_nocal = ind_nan | ind_zero
        raw_cal = np.array([rc[0] for rc in raw_cal])
        raw_cal[ind_nocal==True] = 1

        return (
            raw_cal, 
            raw_back,
            raw_data,
            raw_wvl,
            int_time,
            int_time_t0,
            DarkPixelStart,
            DarkPixelStop,
            )