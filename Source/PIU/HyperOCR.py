# maths
import numpy as np

# python packages
from copy import deepcopy
import calendar
from inspect import currentframe, getframeinfo

# typing
from typing import Union, Any, Optional
from collections import OrderedDict

# Source files
from Source.Utilities import Utilities
from Source.HDFGroup import HDFGroup

# PIU files
from Source.PIU.BaseInstrument import BaseInstrument


class HyperOCR(BaseInstrument):

    def __init__(self):
        super().__init__()
        self.name = "HyperOCR"

    def lightDarkStats(self, grp: dict[str: HDFGroup], XSlice: dict[str: OrderedDict], sensortype: str) -> dict[str: Union[np.array, dict]]:
        """
        Seabird HyperOCR method for retrieving ensemble statistics

        :param grp: dictionary with keys 'LIGHT' for light data and 'DARK' for dark data. 
        The dictionary value should be the HDFGroup of the associated data type for the entire RAW file
        :param XSlice: dictionary with keys 'LIGHT' and 'DARK' for respective data types. 
        The dictionary value should be the associated data for the ensemble
        :param sensortype: name of sensortype, i.e. ES, LI or LT
        """
        lightGrp = grp['LIGHT']
        lightSlice = deepcopy(XSlice['LIGHT'])
        darkGrp = grp['DARK']
        darkSlice = deepcopy(XSlice['DARK'])

        if darkGrp.attributes["FrameType"] == "ShutterDark" and darkGrp.getDataset(sensortype):
            darkData = darkSlice['data'] 
        if lightGrp.attributes["FrameType"] == "ShutterLight" and lightGrp.getDataset(sensortype):
            lightData = lightSlice['data']
        
        # check valid data for retireving stats
        if darkGrp is None or lightGrp is None:
            msg = f'No radiometry found for {sensortype}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False
        elif not HyperOCRUtils.check_data(darkData, lightData):
            return False

        # store results locally for speed
        std_light=[]
        std_dark=[]
        ave_light=[]
        ave_dark = []
        std_signal = {}

        # number of replicates for light and dark readings
        N = np.asarray(list(lightData.values())).shape[1]
        Nd = np.asarray(list(darkData.values())).shape[1]
        for i, k in enumerate(lightData.keys()):
            wvl = str(float(k))

            # apply normalisation to the standard deviations used in uncertainty calculations
            if N > 25:  # normal case
                std_light.append(np.std(lightData[k])/np.sqrt(N))
                std_dark.append(np.std(darkData[k])/np.sqrt(Nd) )  # sigma here is essentially sigma**2 so N must sqrt
            elif N > 3:  # few scans, use different statistics
                std_light.append(np.sqrt(((N-1)/(N-3))*(np.std(lightData[k]) / np.sqrt(N))**2))
                std_dark.append(np.sqrt(((Nd-1)/(Nd-3))*(np.std(darkData[k]) / np.sqrt(Nd))**2))
            else:
                msg = "too few scans to make meaningful statistics"
                print(msg)
                Utilities.writeLogFile(msg)
                return False

            ave_light.append(np.average(lightData[k]))
            ave_dark.append(np.average(darkData[k]))

            for x in range(N):
                try:
                    lightData[k][x] -= darkData[k][x]
                except IndexError as err:
                    msg = f"Light/Dark indexing error PIU.HypperOCR: {err}"
                    print(msg)
                    Utilities.writeLogFile(msg)
                    return False
            

            signalAve = np.average(lightData[k])

            # Normalised signal standard deviation =
            if signalAve:
                std_signal[wvl] = pow((pow(std_light[i], 2) + pow(std_dark[i], 2))/pow(signalAve, 2), 0.5)
            else:
                std_signal[wvl] = 0.0

        return dict(
            ave_Light=np.array(ave_light),
            ave_Dark=np.array(ave_dark),
            std_Light=np.array(std_light),
            std_Dark=np.array(std_dark),
            std_Signal=std_signal,
            )  # output as dictionary for use in ProcessL2/PIU

    def FRM(self, node, uncGrp, raw_grps, raw_slices, stats, newWaveBands):
        pass


class HyperOCRUtils:
    """
    static method to hold utility methods specific to HyperOCR systems
    """
    def __init__(self):
        pass

    @staticmethod
    def check_data(dark, light):
        msg = None
        if (dark is None) or (light is None):
            msg = f'Dark Correction, dataset not found: {dark} , {light}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False

        if Utilities.hasNan(light):
            frameinfo = getframeinfo(currentframe())
            msg = f'found NaN {frameinfo.lineno}'

        if Utilities.hasNan(dark):
            frameinfo = getframeinfo(currentframe())
            msg = f'found NaN {frameinfo.lineno}'
        if msg:
            print(msg)
            Utilities.writeLogFile(msg)
        return True

    @staticmethod
    def darkToLightTimer(rawGrp, sensortype):
        darkGrp = rawGrp['DARK']
        lightGrp = rawGrp['LIGHT']

        if darkGrp.attributes["FrameType"] == "ShutterDark" and darkGrp.getDataset(sensortype):
            darkData = darkGrp.getDataset(sensortype)
            darkDateTime = darkGrp.getDataset("DATETIME")
        if lightGrp.attributes["FrameType"] == "ShutterLight" and lightGrp.getDataset(sensortype):
            lightData = lightGrp.getDataset(sensortype)
            lightDateTime = lightGrp.getDataset("DATETIME")

        if darkGrp is None or lightGrp is None:
            msg = f'No radiometry found for {sensortype}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False
        elif not HyperOCRUtils.check_data(darkData, lightData):
            return False

        newDarkData = HyperOCRUtils.LightDarkInterp(lightData, lightDateTime, darkData, darkDateTime)
        if isinstance(newDarkData, bool):
            return False
        else:
            rawGrp['DARK'].datasets[sensortype].data = newDarkData
            rawGrp['DARK'].datasets[sensortype].datasetToColumns()
            return True

    @staticmethod
    def LightDarkInterp(lightData, lightTimer, darkData, darkTimer):
        # Interpolate Dark Dataset to match number of elements as Light Dataset
        newDarkData = np.copy(lightData.data)
        for k in darkData.data.dtype.fields.keys():  # For each wavelength
            x = np.copy(darkTimer.data).tolist()     # darktimer
            y = np.copy(darkData.data[k]).tolist()   # data at that band over time
            new_x = lightTimer.data                  # lighttimer

            if len(x) < 3 or len(y) < 3 or len(new_x) < 3:
                msg = "**************Cannot do cubic spline interpolation, length of datasets < 3"
                print(msg)
                Utilities.writeLogFile(msg)
                return False
            if not Utilities.isIncreasing(x):
                msg = "**************darkTimer does not contain strictly increasing values"
                print(msg)
                Utilities.writeLogFile(msg)
                return False
            if not Utilities.isIncreasing(new_x):
                msg = "**************lightTimer does not contain strictly increasing values"
                print(msg)
                Utilities.writeLogFile(msg)
                return False

            if len(x) >= 3:
                # Because x is now a list of datetime tuples, they'll need to be converted to Unix timestamp values
                xTS = [calendar.timegm(xDT.utctimetuple()) + xDT.microsecond / 1E6 for xDT in x]
                newXTS = [calendar.timegm(xDT.utctimetuple()) + xDT.microsecond / 1E6 for xDT in new_x]

                newDarkData[k] = Utilities.interp(xTS,y,newXTS, fill_value=np.nan)

                for val in newDarkData[k]:
                    if np.isnan(val):
                        frameinfo = getframeinfo(currentframe())
                        msg = f'found NaN {frameinfo.lineno}'
            else:
                msg = '**************Record too small for splining. Exiting.'
                print(msg)
                Utilities.writeLogFile(msg)
                return False

        if Utilities.hasNan(darkData):
            frameinfo = getframeinfo(currentframe())
            msg = f'found NaN {frameinfo.lineno}'
            print(msg)
            Utilities.writeLogFile(msg)
            return False

        return newDarkData
