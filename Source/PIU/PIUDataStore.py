# linting
from typing import Optional, Union

# math
import numpy as np

# Source
from Source.ConfigFile import ConfigFile
from Source.Utilities import Utilities
from Source.HDFRoot import HDFRoot
from Source.HDFGroup import HDFGroup
from Source.HDFDataset import HDFDataset

# contains class to read and store input uncertainties for PIU

class PIUDataStore:
    sensors: list = ['ES', 'LI', 'LT']

    def __init__(self, root: HDFRoot, input: HDFGroup):
        """ class which contains methods that provide digestable uncertainties to classes in PIU """
        self.uncs:      dict = {s: [] for s in self.sensors}
        self.coeff:     dict = {s: [] for s in self.sensors}
        self.cal_level: int = ConfigFile.settings["fL1bCal"]

        self.cal_start: int = None
        self.cal_stop:  int = None

        self.ind_rad_wvl: dict = {s: [] for s in self.sensors}
        self.nan_mask:    np.array = None

        instrument = ConfigFile.settings['SensorType'].lower()  # get instrument type
        if self.cal_level == 2:
            [self.readCalClassBased(input, sensor, instrument) for sensor in self.sensors]
        elif instrument == 'seabird':
            [self.readCalFactory(root, input, sensor) for sensor in self.sensors]
        else:
            msg = "TriOS/Dalec factory uncertainties not implemented"
            Utilities.writeLogFile(msg)
            print(msg)
            raise NotImplementedError

        [self.read_uncertainties(input, sensor) for sensor in self.sensors]
    
    def readCalClassBased(self, inpt: HDFGroup, s: str, i_type: str) -> None:
        radcal = self.extract_unc_from_grp(inpt, f"{s}_RADCAL_CAL")
        ind_rad_wvl = (np.array(radcal.columns['1']) > 0)  # where radcal wvls are available
        
        corr_factor = 10 if (i_type == "trios" or i_type == "sorad") else 1 # Convert TriOS mW/m2/nm to uW/cm^2/nm
        self.coeff['cal'][s] = np.asarray(list(radcal.columns['2']))[ind_rad_wvl] / corr_factor
        self.uncs['cal'][s] = np.asarray(list(radcal.columns['3']))[ind_rad_wvl]

        self.ind_rad_wvl['s'] = ind_rad_wvl

    def readCalFactory(self, node: HDFRoot, inpt: HDFGroup, s: str) -> None:
        radcal = self.extract_unc_from_grp(inpt, f"{s}_RADCAL_UNC")
        ind_rad_wvl = (np.array(radcal.columns['wvl']) > 0)  # all radcal wvls should be available from sirrex
        # read cal start and cal stop for shaping stray-light class based uncertainties
        self.cal_start = int(node.attributes['CAL_START'])
        self.cal_stop = int(node.attributes['CAL_STOP'])

        self.uncs['cal'], self.coeff['cal'] = self.extract_factory_cal(node, radcal, s)  # populates dicts with calibration
        self.ind_rad_wvl['s'] = ind_rad_wvl

    def read_uncertainties(self, inpt: HDFGroup, s: str) -> None:
        self.uncs['stab'][s] = self.extract_unc_from_grp(inpt, f"{s}_STABDATA_CAL", '1')
        self.uncs['stray'][s] = self.extract_unc_from_grp(inpt, f"{s}_STRAYDATA_CAL", '1')
        self.clipSL(s)

        self.uncs['nlin'][s] = self.extract_unc_from_grp(grp=inpt, name=f"{s}_NLDATA_CAL", col_name='1')
        self.uncs['ct'][s] = self.extract_unc_from_grp(grp=inpt, name=f"{s}_TEMPDATA_CAL", col_name=f'{s}_TEMPERATURE_UNCERTAINTIES')

        if "ES" in s.upper():
            self.uncs['cos'][s] = self.extract_unc_from_grp(grp=inpt, name=f"{s}_POLDATA_CAL", col_name='1')
        else:
            self.uncs['pol'][s] = self.extract_unc_from_grp(grp=inpt, name=f"{s}_POLDATA_CAL", col_name='1')
        
        self.nan_mask = np.where(any([(u[s] <= 0) for u in self.uncs]))

    ## UTILITIES ##

    def clipSL(self, s: str) -> None:
        start = self.cal_start
        stop = self.cal_stop
        ind_wvl = self.ind_rad_wvl[s]

        if (ind_wvl is not None) and (len(ind_wvl) == len(self.uncs['stray'][s])):
            self.uncs['stray'][s] = self.uncs['stray'][s][ind_wvl]
        elif (start is not None) and (stop is not None):
            self.uncs['stray'][s] = self.uncs['stray'][s][start:stop + 1]
        else:
            msg = "cannot mask straylight"
            print(msg)  # to cover for potential coding errors, should not be hit in normal use

    @staticmethod
    def extract_factory_cal(node: HDFGroup, radcal: np,array, s: str) -> tuple[np.array, np.array]:
        """

        :param node: HDF root - full HDF file
        :param radcal: HDF group containing radiometric calibration
        :param s: dict key to append data to cCal and cCoef
        :param cCal: dict for storing calibration
        :param cCoef: dict for storing calibration coeficients 
        """
        from os import path
        from Source import PATH_TO_CONFIG
        from Source.CalibrationFileReader import CalibrationFileReader
        from Source.ProcessL1b_FactoryCal import ProcessL1b_FactoryCal

        cal = np.asarray(list(radcal.columns['unc']))
        calFolder = path.splitext(ConfigFile.filename)[0] + "_Calibration"
        calPath = path.join(PATH_TO_CONFIG, calFolder)
        calibrationMap = CalibrationFileReader.read(calPath)

        if ConfigFile.settings['SensorType'].lower() == "dalec":
            _, coef = ProcessL1b_FactoryCal.extract_calibration_coeff_dalec(calibrationMap, s)
        else:    
            _, coef = ProcessL1b_FactoryCal.extract_calibration_coeff(node, calibrationMap, s)

        return cal, coef
    
    @staticmethod
    def extract_unc_from_grp(grp: HDFGroup, name: str, col_name: Optional[str] = None) -> Union[np.array, HDFDataset]:
        """

        :param grp: HDF group to take dataset from
        :param name: name of dataset
        :param col_name: name of column to extract unc from
        """
        ds = grp.getDataset(name)
        ds.datasetToColumns()
        if col_name is not None:
            return np.asarray(list(ds.columns[col_name]))
        else:
            return ds