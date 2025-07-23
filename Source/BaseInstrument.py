# Import Python packages
import numpy as np
import scipy as sp
import pandas as pd
import copy
import warnings
from datetime import datetime
from collections import OrderedDict
from decimal import Decimal
from abc import ABC, abstractmethod
from typing import Union, Optional, Any

# NPL packages
import comet_maths as cm

# Source files
from Source import PATH_TO_CONFIG
from Source.Utilities import Utilities
from Source.ConfigFile import ConfigFile
from Source.HDFRoot import HDFRoot
from Source.HDFGroup import HDFGroup
from Source.HDFDataset import HDFDataset
from Source.Uncertainty_Analysis import Propagate
from Source.Weight_RSR import Weight_RSR
from Source.CalibrationFileReader import CalibrationFileReader
from Source.ProcessL1b_FactoryCal import ProcessL1b_FactoryCal
from Source.Uncertainty_Visualiser import UncertaintyGUI


class BaseInstrument(ABC):  # Inheriting ABC allows for more function decorators which exist to give warnings to coders.
    """Base class for instrument uncertainty analysis. Abstract methods are utilised where appropriate. The core idea is
    to reuse as much code as possible whilst making it simpler to add functionality through the addition of child
    classes"""

    # variable placed here will be made available to all instances of Instrument class. Varname preceded by '_'
    # to indicate privacy, this should NOT be changed at runtime
    _SATELLITES: dict = {
        "S3A": {"name": "Sentinel3A", "config": "bL2WeightSentinel3A", "Weight_RSR": Weight_RSR.Sentinel3Bands()},
        "S3B": {"name": "Sentinel3B", "config": "bL2WeightSentinel3B", "Weight_RSR": Weight_RSR.Sentinel3Bands()},
        "MOD-A": {"name": "MODISA", "config": "bL2WeightMODISA", "Weight_RSR": Weight_RSR.MODISBands()},
        "MOD-T": {"name": "MODIST", "config": "bL2WeightMODIST", "Weight_RSR": Weight_RSR.MODISBands()},
        "VIIRS-N": {"name": "VIIRSN", "config": "bL2WeightVIIRSN", "Weight_RSR": Weight_RSR.VIIRSBands()},
        "VIIRS-J": {"name": "VIIRSJ", "config": "bL2WeightVIIRSJ", "Weight_RSR": Weight_RSR.VIIRSBands()},
    }  # list of avaialble sensors with their config file names a name for the xUNC key and associated Weight_RSR bands

    def __init__(self):
        # use this to switch the straylight correction method -> FOR UNCERTAINTY PROPAGATION ONLY <- between SLAPER and
        # ZONG. Not added to config file settings because this isn't intended for the end user.
        self.sl_method: str = 'ZONG'

    @abstractmethod
    def lightDarkStats(self, grp: Union[HDFGroup, list], slice: list, sensortype: str) -> dict[np.array]:
        """
        method to return the noise (std before and after light-dark substitution) and averages for light and dark data. 
        Refer to D-10 figure-8, Eq-10 for Radiance and figure-9, Eq-11 for Irradiance. Both figures and equations indicate 
        signal as "S" with std Dark/Light being DN_dark/DN_light respectively. 

        :param grp: HDFGroup representing the sensor specific data
        :param slice: Ensembled sensor data
        :param sensortype: sensor name

        :return:
        """
        # abstract method indicates the requirement for all child/derived classes to have a lightDarkStats method, this will be
        # sensor specific and is required for generateSensorStats. For Dalec (or other sensors) it must be a function
        # that outputs a dictionary containing:
        # {
        # "ave_Light": averaged light data,
        # "ave_Dark": averaged dark data,
        # "std_Light": standard deviation from the mean of light data,
        # "std_Dark": standard deviation from the mean of dark data,
        # "std_Signal" standard deviation from the mean of the instrument signal,
        # }
        # all standard deviations are divided by root N (number of scans) to become standard deviation from the mean.
        pass

    def generateSensorStats(self, InstrumentType: str, rawData: dict, rawSlice: dict, newWaveBands: np.array
                            ) -> dict[str: np.array]:
        """
        Generate Sensor Stats calls lightDarkStats for a given instrument. Once sensor statistics are known, they are 
        interpolated to common wavebands to match the other L1B sensor inputs Es, Li, & Lt.

        :return: dictionary of statistics used later in the processing pipeline. Keys are:
        [ave_Light, ave_Dark, std_Light, std_Dark, std_Signal]
        """
        output = {}  # used tp store standard deviations and averages as a function return for generateSensorStats
        types = ['ES', 'LI', 'LT']
        for sensortype in types:
           if InstrumentType.lower() == "trios" or InstrumentType.lower() == "sorad":
                # filter nans
                self.apply_NaN_Mask(rawSlice[sensortype]['data'])
                # RawData is the full group - this is used to get a few attributes only
                # rawSlice is the ensemble 'slice' of raw data currently to be evaluated
                #  todo: check the shape and that there are no nans or infs
                output[sensortype] = self.lightDarkStats(
                    copy.deepcopy(rawData[sensortype]), copy.deepcopy(rawSlice[sensortype]), sensortype
                )
                # copy.deepcopy ensures RAW data is unchanged for FRM uncertainty generation.
           elif InstrumentType.lower() == "dalec":
                # RawData is the full group - this is used to get a few attributes only
                # rawSlice is the ensemble 'slice' of raw data currently to be evaluated
                #  todo: check the shape and that there are no nans or infs
                output[sensortype] = self.lightDarkStats(
                    copy.deepcopy(rawData[sensortype]), copy.deepcopy(rawSlice[sensortype]), sensortype
                )
                # copy.deepcopy ensures RAW data is unchanged for FRM uncertainty generation.
           elif InstrumentType.lower() == "seabird":
                # rawData here is the group, passed along only for the purpose of
                # confirming "FrameTypes", i.e., ShutterLight or ShutterDark. Calculations
                # are performed on the Slice.
                # output contains:
                # ave_Light: (array 1 x number of wavebands)
                # ave_Dark: (array 1 x number of wavebands)
                # std_Light: (array 1 x number of wavebands)
                # std_Dark: (array 1 x number of wavebands)
                # std_Signal: OrdDict by wavebands: sqrt( (std(Light)^2 + std(Dark)^2)/ave(Light)^2 )

                # filter nans
                # this should work because of the interpolation, however I cannot test this as I do not have seabird
                # data with NaNs
                # self.apply_NaN_Mask(rawSlice[sensortype]['LIGHT'])
                # self.apply_NaN_Mask(rawSlice[sensortype]['DARK'])
                output[sensortype] = self.lightDarkStats(
                    [rawData[sensortype]['LIGHT'],
                    rawData[sensortype]['DARK']],
                    [rawSlice[sensortype]['LIGHT'],
                    rawSlice[sensortype]['DARK']],
                    sensortype
                )
           if not output[sensortype]:
                msg = "Could not generate statistics for the ensemble"
                print(msg)
                return False

        # interpolate std Signal to common wavebands - taken from L2 ES group: ProcessL2.py L1352
        for stype in types:
            try:
                output[stype]['std_Signal_Interpolated'] = self.interp_common_wvls(
                    output[stype]['std_Signal'],
                    np.asarray(list(output[stype]['std_Signal'].keys()), dtype=float),
                    newWaveBands,
                    return_as_dict=True)
                    # this interpolation is giving an array back of a slightly different size in the new wave bands
            except IndexError as err:
                msg = "Unable to parse statistics for the ensemble, possibly too few scans."
                print(msg)
                Utilities.writeLogFile(msg)
                return False
        #print("generateSensorStats: output(stats)")
        #print(output)
        return output

    def read_uncertainties(self, node, uncGrp, cCal, cCoef, cStab, cLin, cStray, cT, cPol, cCos) -> Optional[np.array]:
        """
        reads the uncertainties from the HDF file, must return indicated raw bands, i.e. which bands we have uncertainty 
        values saved in the cal/char files.
        
        :param node: HDFRoot of input HDF is required to retrieve calibration file start and stop for slicing straylight
        :param uncGrp: HDFGroup Uncertainties from HDF is required to retrieve uncertainties
        :param cCal: dict to contain calibration
        :param cCoef: dict to contain calibration coefficient uncertainty
        :param cStab: dict to contain stability information
        :param cLin: dict to contain non-linearity information
        :param cStray: dict to contain straylight information
        :param cT: dict to contain temperature correction information
        :param cPol: dict to contain polarisation information
        :param cCos: dict to contain cosine response information
        """

        for s in ["ES", "LI", "LT"]:  # s for sensor type
            cal_start = None
            cal_stop = None
            if ConfigFile.settings["fL1bCal"] == 1 and ConfigFile.settings['SensorType'].lower() == "seabird":
                radcal = self.extract_unc_from_grp(uncGrp, f"{s}_RADCAL_UNC")
                ind_rad_wvl = (np.array(radcal.columns['wvl']) > 0)  # all radcal wvls should be available from sirrex
                # read cal start and cal stop for shaping stray-light class based uncertainties
                cal_start = int(node.attributes['CAL_START'])
                cal_stop = int(node.attributes['CAL_STOP'])

                self.extract_factory_cal(node, radcal, s, cCal, cCoef)  # populates dicts with calibration

                        
            #elif ConfigFile.settings["fL1bCal"] == 1 and ConfigFile.settings['SensorType'].lower() == "dalec":
            #    radcal = self.extract_unc_from_grp(uncGrp, f"{s}_RADCAL_UNC")
            #    ind_rad_wvl = (np.array(radcal.columns['wvl']) > 0)  # all radcal wvls should be available from sirrex
            #    self.extract_factory_cal(node, radcal, s, cCal, cCoef)  # populates dicts with calibration
            
            elif ConfigFile.settings["fL1bCal"] == 2:  # class-Based
                radcal = self.extract_unc_from_grp(uncGrp, f"{s}_RADCAL_CAL")
                ind_rad_wvl = (np.array(radcal.columns['1']) > 0)  # where radcal wvls are available

                # ensure correct units are used for uncertainty calculation
                if ConfigFile.settings['SensorType'].lower() == "trios" or ConfigFile.settings['SensorType'].lower() == "sorad":
                    # Convert TriOS mW/m2/nm to uW/cm^2/nm
                    cCoef[s] = np.asarray(list(radcal.columns['2']))[ind_rad_wvl] / 10
                elif ConfigFile.settings['SensorType'].lower() == "seabird":
                    cCoef[s] = np.asarray(list(radcal.columns['2']))[ind_rad_wvl]
                cCal[s] = np.asarray(list(radcal.columns['3']))[ind_rad_wvl]

            else:
                msg = "TriOS/Dalec factory uncertainties not implemented"
                Utilities.writeLogFile(msg)
                print(msg)
                return False,False

            cStab[s] = self.extract_unc_from_grp(uncGrp, f"{s}_STABDATA_CAL", '1')

            cStray[s] = self.extract_unc_from_grp(uncGrp, f"{s}_STRAYDATA_CAL", '1')
            if (ind_rad_wvl is not None) and (len(ind_rad_wvl) == len(cStray[s])):
                cStray[s] = cStray[s][ind_rad_wvl]
            elif (cal_start is not None) and (cal_stop is not None):
                cStray[s] = cStray[s][cal_start:cal_stop + 1]
            else:
                # to cover for potential coding errors, should not be hit in normal use
                msg = "cannot mask straylight"
                print(msg)
                return False,False

            cLin[s] = self.extract_unc_from_grp(grp=uncGrp, name=f"{s}_NLDATA_CAL", col_name='1')

            # temp uncertainties calculated at L1AQC
            cT[s] = self.extract_unc_from_grp(grp=uncGrp,
                                              name=f"{s}_TEMPDATA_CAL",
                                              col_name=f'{s}_TEMPERATURE_UNCERTAINTIES')

            # temporary fix angular for ES is written as ES_POL
            if "ES" in s:
                cCos[s] = self.extract_unc_from_grp(grp=uncGrp, name=f"{s}_POLDATA_CAL", col_name='1')
            else:
                cPol[s] = self.extract_unc_from_grp(grp=uncGrp, name=f"{s}_POLDATA_CAL", col_name='1')

            nan_mask = np.where((cStab[s] <= 0) | (cStray[s] <= 0) | (cLin[s] <= 0) | (cT[s] <= 0) |
                                (self.extract_unc_from_grp(grp=uncGrp, name=f"{s}_POLDATA_CAL", col_name='1') <= 0))

        return ind_rad_wvl, nan_mask

    def ClassBased(self, node: HDFRoot, uncGrp: HDFGroup, stats: dict[str, np.array]) -> Union[dict[str, dict], bool]:
        """
        Propagates class based uncertainties for all instruments. If no calibration uncertainties are available will use Sirrex-7 
        to propagate uncertainties in the SeaBird Case. See D-10 secion 5.3.1.

        :param node: HDFRoot containing all L1BQC data
        :param uncGrp: HDFGroup containing raw uncertainties
        :param stats: output of PIU.py BaseInstrument.generateSensorStats

        :return: dictionary of instrument uncertainties [Es uncertainty, Li uncertainty, Lt uncertainty]
        alternatively errors in processing will return False for context management purposes.
        """

        # create object for running uncertainty propagation, M means number of monte carlo draws
        Prop_Instrument_CB = Propagate(M=100, cores=0)  # Propagate_Instrument_Uncertainty_ClassBased

        # initialise dicts for error sources
        cCal = {}
        cCoef = {}
        cStab = {}
        cLin = {}
        cStray = {}
        cT = {}
        cPol = {}
        cCos = {}

        ind_rad_wvl, nan_mask = self.read_uncertainties(
            node,
            uncGrp,
            cCal=cCal,
            cCoef=cCoef,
            cStab=cStab,
            cLin=cLin,
            cStray=cStray,
            cT=cT,
            cPol=cPol,
            cCos=cCos
        )
        if isinstance(ind_rad_wvl, bool):
            return False

        ones = np.ones_like(cCal['ES'])  # array of ones with correct shape.

        means = [stats['ES']['ave_Light'], stats['ES']['ave_Dark'],
                 stats['LI']['ave_Light'], stats['LI']['ave_Dark'],
                 stats['LT']['ave_Light'], stats['LT']['ave_Dark'],
                 cCoef['ES'], cCoef['LI'], cCoef['LT'],
                 ones, ones, ones,
                 ones, ones, ones,
                 ones, ones, ones,
                 ones, ones, ones,
                 ones, ones, ones
                 ]

        uncertainties = [stats['ES']['std_Light'], stats['ES']['std_Dark'],
                         stats['LI']['std_Light'], stats['LI']['std_Dark'],
                         stats['LT']['std_Light'], stats['LT']['std_Dark'],
                         cCal['ES'] * cCoef['ES'] / 200,
                         cCal['LI'] * cCoef['LI'] / 200,
                         cCal['LT'] * cCoef['LT'] / 200,
                         cStab['ES'], cStab['LI'], cStab['LT'],
                         cLin['ES'], cLin['LI'], cLin['LT'],
                         np.array(cStray['ES']) / 100,
                         np.array(cStray['LI']) / 100,
                         np.array(cStray['LT']) / 100,
                         np.array(cT['ES']), np.array(cT['LI']), np.array(cT['LT']),
                         np.array(cPol['LI']), np.array(cPol['LT']), np.array(cCos['ES'])
                         ]

        # generate uncertainties using Monte Carlo Propagation object
        es_unc, li_unc, lt_unc = Prop_Instrument_CB.propagate_Instrument_Uncertainty(means, uncertainties)

        # NOTE: Debugging check
        is_negative = np.any([ x < 0 for x in means])
        if is_negative:
            print('WARNING: Negative uncertainty potential')
        is_negative = np.any([ x < 0 for x in uncertainties])
        if is_negative:
            print('WARNING: Negative uncertainty potential')
        if any(es_unc < 0) or any(li_unc < 0) or any(lt_unc < 0):
            print('WARNING: Negative uncertainty potential')

        es, li, lt = Prop_Instrument_CB.instruments(*means)

        # plot class based L1B uncertainties
        rad_cal_str = "ES_RADCAL_CAL" if "ES_RADCAL_CAL" in uncGrp.datasets.keys() else "ES_RADCAL_UNC"
        cal_col_str = "1" if "ES_RADCAL_CAL" in uncGrp.datasets.keys() else "wvl"
        if ConfigFile.settings['bL2UncertaintyBreakdownPlot']:
            #       NOTE: For continuous, autonomous acquisition (e.g. SolarTracker, pySAS, SoRad, DALEC), stations are
            #       only associated with specific spectra during times that intersect with station designation in the
            #       Ancillary file. If station extraction is performed in L2, then the resulting HDF will have only one
            #       unique station designation, though that may include multiple ensembles, depending on how long the ship
            #       was on station. - DA
            acqTime = datetime.strptime(node.attributes['TIME-STAMP'], '%a %b %d %H:%M:%S %Y')
            cast = f"{type(self).__name__}_{acqTime.strftime('%Y%m%d%H%M%S')}"

            # the breakdown plots must calculate uncertainties separately from the main processor, will incur additional
            # computational overheads
            p_unc = UncertaintyGUI(Prop_Instrument_CB)
            p_unc.pie_plot_class(
                means,
                uncertainties,
                dict(
                    ES=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str]),
                    LI=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str]),
                    LT=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str])
                ),
                cast,
                node.getGroup("ANCILLARY")
            )
            p_unc.plot_class(
                means,
                uncertainties,
                dict(
                    ES=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str]),
                    LI=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str]),
                    LT=np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str])
                ),
                cast
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            # convert to relative in order to avoid a complex unit conversion process in ProcessL2.

            ES_unc = es_unc / np.abs(es)
            LI_unc = li_unc / np.abs(li)
            LT_unc = lt_unc / np.abs(lt)


        # interpolation step - bringing uncertainties to common wavebands from radiometric calibration wavebands.
        data_wvl = np.asarray(list(stats['ES']['std_Signal_Interpolated'].keys()),
                              dtype=float)
    
        es_Unc = self.interp_common_wvls(ES_unc,
                                         np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                                     dtype=float)[ind_rad_wvl],
                                         data_wvl,
                                         return_as_dict=True
                                         )
        li_Unc = self.interp_common_wvls(LI_unc,
                                         np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                                  dtype=float)[ind_rad_wvl],
                                         data_wvl,
                                         return_as_dict=True
                                         )
        lt_Unc = self.interp_common_wvls(LT_unc,
                                         np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                                  dtype=float)[ind_rad_wvl],
                                         data_wvl,
                                         return_as_dict=True
                                         )
        
        # return uncertainties to ProcessL2 as dictionary - will update xUnc dict with new uncs propagated to L1B
        return dict(
            esUnc=es_Unc,
            liUnc=li_Unc,
            ltUnc=lt_Unc,
            valid_pixels=nan_mask,
        )

    def FRM(self, node: HDFRoot, uncGrp: HDFGroup, raw_grps: dict[str, HDFGroup], raw_slices: dict[str, np.array],
            stats: dict, newWaveBands: np.array) -> dict[str, np.array]:
        """
        Propagates instrument uncertainties with corrections (except polarisation) if full characterisation available - see D-10 section 5.3.1
        
        :param node: HDFRoot of L1BQC data for processing
        :param uncGrp: HDFGroup of uncertainty budget
        :param raw_grps: dictionary of raw data groups
        :param raw_slices: dictionary of sliced data for specific sensors
        :param stats: standard deviation and averages for Light, Dark and Light-Dark signal
        :param newWaveBands: wavelength subset for interpolation

        :return: output FRM uncertainties
        """

        unc_dict = {}
        for sensortype in ['ES', 'LI', 'LT']:

            # straylight
            unc_dict[f"mZ_unc_{sensortype}"] = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_STRAYDATA_UNCERTAINTY").data))
            # temperature
            unc_dict[f"Ct_unc_{sensortype}"] = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_TEMPDATA_CAL").data[1:].transpose().tolist())[5])
            # Radcal Cal S1/S2
            unc_dict[f'S1_unc_{sensortype}'] = (pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['7'])[1:]
            unc_dict[f'S2_unc_{sensortype}'] = (pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['9'])[1:]
            # Stability
            # unc_dict[f'stab_{sensortype}'] = self.extract_unc_from_grp(uncGrp, f"{sensortype}_STABDATA_CAL", '1')  # class based method
            # Nlin
            # if I remove uncertainties in S1/S2 then I necessarily remove the Nlin unc!

            # Lamp_cal - part of radcal corr
            LAMP = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_LAMP").data)['2']) / 10  # div by 10
            unc_dict[f'lamp_{sensortype}'] = (np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_LAMP").data)['3'])/100)*LAMP
            
            if sensortype == 'ES':
                # Cosine
                coserror = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").data))[1:, 2:]
                coserror_90 = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR_AZ90").data))[1:, 2:]
                unc_dict['cos_unc'] = (np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY").data))[1:, 2:] / 100) * np.abs(coserror)
                unc_dict['cos90_unc'] = (np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY_AZ90").data))[1:, 2:] / 100) * np.abs(coserror_90)
            else:
                # Polarisation
                # read pol uncertainties and interpolate to radcal wavebands
                radcal_wvl = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['1'][1:].tolist())
                pol = uncGrp.getDataset(f"CLASS_RAMSES_{sensortype}_POLDATA_CAL")
                pol.datasetToColumns()
                x = pol.columns['0']
                y = pol.columns['1']
                y_new = np.interp(radcal_wvl, x, y)
                pol.columns['0'] = radcal_wvl
                pol.columns['1'] = y_new
                unc_dict[f'pol_unc_{sensortype}'] = np.asarray(list(pol.columns['1']))
                
                # Panel - part of radcal corr
                PANEL = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_PANEL").data)['2'])
                unc_dict[f'unc_PANEL_{sensortype}'] = (np.asarray(
                    pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_PANEL").data)['3'])/100)*PANEL

        return self.processFRM(node, uncGrp, unc_dict, raw_grps, raw_slices, stats, newWaveBands)

    @abstractmethod
    def processFRM(self, node, uncGrp, uncDict, raw_grps, raw_slices, stats, newWaveBands) -> dict[str, Any]:
        """
        FRM regime propagation instrument uncertainties, see D10 section 5.3.2 for more information.
        :param node: HDFRoot containing entire HDF file
        :param uncGrp: HDFGroup containing uncertainties from HDF file
        :param raw_grps: raw data dictionary containing Es, Li, & Lt as HDFGroups
        :param raw_slices: sliced raw data dictionary containing Es, Li, & Lt as np.arrays
        :param stats: not required for TriOS specific processing, set to None at start of method
        :param newWaveBands: common wavebands for interpolation of output
        """
        pass
    
    ## L2 uncertainty Processing
    def FRM_L2(self, rhoScalar: float, rhoVec: np.array, rhoDelta: np.array, waveSubset: np.array,
               xSlice: dict[str, np.array]) -> dict[str, np.array]:
        """
        Propagates Lw and Rrs uncertainties if full characterisation available - see D-10 5.3.1

        :param rhoScalar: rho input if Mobley99 or threeC rho is used
        :param rhoVec: rho input if Zhang17 rho is used
        :param rhoDelta: uncertainties associated with rho
        :param waveSubset: wavelength subset for any band convolution (and sizing rhoScalar if used)
        :param xSlice: Dictionary of input radiance, raw_counts, standard deviations etc.

        :return: dictionary of output uncertainties that are generated

        """
        # organise data
        # cut data down to wavelengths where rho values exist -- should be no change for M99
        esSampleXSlice = np.asarray([{key: sample for key, sample in
                                      xSlice['esSample'][i].items() if float(key) in waveSubset}
                                     for i in range(len(xSlice['esSample']))])
        liSampleXSlice = np.asarray([{key: sample for key, sample in
                                      xSlice['liSample'][i].items() if float(key) in waveSubset}
                                     for i in range(len(xSlice['liSample']))])
        ltSampleXSlice = np.asarray([{key: sample for key, sample in
                                      xSlice['ltSample'][i].items() if float(key) in waveSubset}
                                     for i in range(len(xSlice['ltSample']))])

        # Get rho from scalar or vector
        if rhoScalar is not None:  # make rho a constant array if scalar
            rho = np.ones(len(waveSubset))*rhoScalar  # convert rhoScalar to the same dims as other values/Uncertainties
        else:
            rho = np.asarray(list(rhoVec.values()), dtype=float)

        # Get rhoDelta from scalar or vector
        if not hasattr(rhoDelta, '__len__'):  # Not an array (e.g. list or np.array)
            rhoDelta = np.ones(len(waveSubset)) * rhoDelta  # convert rhoDelta to the same dims as other values/Uncertainties

        # initialise punpy propagation object
        mdraws = esSampleXSlice.shape[0]  # keep no. of monte carlo draws consistent
        Propagate_L2_FRM = Propagate(mdraws, cores=1)  # punpy.MCPropagation(mdraws, parallel_cores=1)

        # get sample for rho
        rhoSample = cm.generate_sample(mdraws, rho, rhoDelta, "syst")

        # initialise lists to store uncertainties per replicate

        esSample = np.asarray([[i[0] for i in k.values()] for k in esSampleXSlice])  # recover original shape of samples
        liSample = np.asarray([[i[0] for i in k.values()] for k in liSampleXSlice])
        ltSample = np.asarray([[i[0] for i in k.values()] for k in ltSampleXSlice])

        # no uncertainty in wavelengths
        sample_wavelengths = cm.generate_sample(mdraws, np.array(waveSubset), None, None)
        # Propagate_L2_FRM is a Propagate object defined in Uncertainty_Analysis, this stores a punpy MonteCarlo
        # Propagation object (punpy.MCP) as a class member variable Propagate.MCP. We can therefore use this to get to
        # the punpy.MCP namespace to access punpy specific methods such as 'run_samples'. This has a memory saving over
        # making a separate object for running these methods.
        sample_Lw = Propagate_L2_FRM.MCP.run_samples(Propagate.Lw_FRM, [ltSample, rhoSample, liSample])
        sample_Rrs = Propagate_L2_FRM.MCP.run_samples(Propagate.Rrs_FRM, [ltSample, rhoSample, liSample, esSample])

        output = {}

        for s_key in self._SATELLITES.keys():
            output.update(
                self.get_band_outputs_FRM(
                s_key, Propagate_L2_FRM, esSample, liSample, ltSample, rhoSample, sample_wavelengths
                )
            )

        lwDelta = Propagate_L2_FRM.MCP.process_samples(None, sample_Lw)
        rrsDelta = Propagate_L2_FRM.MCP.process_samples(None, sample_Rrs)

        output["rhoUNC_HYPER"] = {str(wvl): val for wvl, val in zip(waveSubset, rhoDelta)}
        output["lwUNC"] = lwDelta  # Multiply by large number to reduce round off error
        output["rrsUNC"] = rrsDelta

        return output

    def ClassBasedL2(self, node, uncGrp, rhoScalar, rhoVec, rhoDelta, waveSubset, xSlice) -> dict:
        """
        Propagates class based uncertainties for all Lw and Rrs. See D-10 secion 5.3.1.

        :param node: HDFRoot which stores L1BQC data
        :param uncGrp: HDFGroup storing the uncertainty budget
        :param rhoScalar: rho input if Mobley99 or threeC rho is used
        :param rhoVec: rho input if Zhang17 rho is used
        :param rhoDelta: uncertainties associated with rho
        :param waveSubset: wavelength subset for any band convolution (and sizing rhoScalar if used)
        :param xSlice: Dictionary of input radiance, raw_counts, standard deviations etc.

        :return: dictionary of output uncertainties that are generated
        """

        Prop_L2_CB = Propagate(M=100, cores=0)
        waveSubset = np.array(waveSubset, dtype=float)  # convert waveSubset to numpy array
        esXstd = xSlice['esSTD_RAW']  # stdevs taken at instrument wavebands (not common wavebands)
        liXstd = xSlice['liSTD_RAW']
        ltXstd = xSlice['ltSTD_RAW']

        if rhoScalar is not None:  # make rho a constant array if scalar
            rho = np.ones(len(list(esXstd.keys()))) * rhoScalar
            rhoUNC = self.interp_common_wvls(np.array(rhoDelta, dtype=float),
                                             waveSubset,
                                             np.asarray(list(esXstd.keys()), dtype=float),
                                             return_as_dict=False)
        else:  # zhang rho needs to be interpolated to radcal wavebands (len must be 255)
            rho = self.interp_common_wvls(np.array(list(rhoVec.values()), dtype=float),
                                          waveSubset,
                                          np.asarray(list(esXstd.keys()), dtype=float),
                                          return_as_dict=False)
            rhoUNC = self.interp_common_wvls(rhoDelta,
                                             waveSubset,
                                             np.asarray(list(esXstd.keys()), dtype=float),
                                             return_as_dict=False)

        # initialise dicts for error sources
        cCal = {}
        cCoef = {}
        cStab = {}
        cLin = {}
        cStray = {}
        cT = {}
        cPol = {}
        cCos = {}

        ind_rad_wvl, nan_mask = self.read_uncertainties(
            node,
            uncGrp,
            cCal=cCal,
            cCoef=cCoef,
            cStab=cStab,
            cLin=cLin,
            cStray=cStray,
            cT=cT,
            cPol=cPol,
            cCos=cCos
        )

        # interpolate to radcal wavebands - check string for radcal group based on factory or class-based processing
        rad_cal_str = "ES_RADCAL_CAL" if "ES_RADCAL_CAL" in uncGrp.datasets.keys() else "ES_RADCAL_UNC"
        cal_col_str = "1" if "ES_RADCAL_CAL" in uncGrp.datasets.keys() else "wvl"
        es = self.interp_common_wvls(np.asarray(list(xSlice['es'].values()), dtype=float).flatten(),
                                     np.asarray(list(xSlice['es'].keys()), dtype=float).flatten(),
                                     np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                              dtype=float)[ind_rad_wvl],
                                     return_as_dict=False)
        li = self.interp_common_wvls(np.asarray(list(xSlice['li'].values()), dtype=float).flatten(),
                                     np.asarray(list(xSlice['li'].keys()), dtype=float).flatten(),
                                     np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                              dtype=float)[ind_rad_wvl],
                                     return_as_dict=False)
        lt = self.interp_common_wvls(np.asarray(list(xSlice['lt'].values()), dtype=float).flatten(),
                                     np.asarray(list(xSlice['lt'].keys()), dtype=float).flatten(),
                                     np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str],
                                              dtype=float)[ind_rad_wvl],
                                     return_as_dict=False)

        ones = np.ones_like(es)

        lw_means = [lt, rho, li,
                    ones, ones,
                    ones, ones,
                    ones, ones,
                    ones, ones,
                    ones, ones,
                    ones, ones]

        lw_uncertainties = [np.abs(np.array(list(ltXstd.values())).flatten() * lt),
                            rhoUNC,
                            np.abs(np.array(list(liXstd.values())).flatten() * li),
                            cCal['LI'] / 200, cCal['LT'] / 200,
                            cStab['LI'], cStab['LT'],
                            cLin['LI'], cLin['LT'],
                            cStray['LI'] / 100, cStray['LI'] / 100,
                            cT['LI'], cT['LI'],
                            cPol['LI'], cPol['LI']]

        lwAbsUnc = Prop_L2_CB.Propagate_Lw_HYPER(lw_means, lw_uncertainties)

        rrs_means = [lt, rho, li, es,
                     ones, ones, ones,
                     ones, ones, ones,
                     ones, ones, ones,
                     ones, ones, ones,
                     ones, ones, ones,
                     ones, ones, ones
                     ]

        rrs_uncertainties = [np.abs(np.array(list(ltXstd.values())).flatten() * lt),
                             rhoUNC,
                             np.abs(np.array(list(liXstd.values())).flatten() * li),
                             np.abs(np.array(list(esXstd.values())).flatten() * es),
                             cCal['ES'] / 200, cCal['LI'] / 200, cCal['LT'] / 200,
                             cStab['ES'], cStab['LI'], cStab['LT'],
                             cLin['ES'], cLin['LI'], cLin['LT'],
                             cStray['ES'] / 100, cStray['LI'] / 100, cStray['LT'] / 100,
                             cT['ES'], cT['LI'], cT['LT'],
                             cPol['LI'], cPol['LT'], cCos['ES']
                             ]

        rrsAbsUnc = Prop_L2_CB.Propagate_RRS_HYPER(rrs_means, rrs_uncertainties)
        #print("rrsAbsUnc")
        #print(rrsAbsUnc)

        # Plot Class based L2 uncertainties
        if ConfigFile.settings['bL2UncertaintyBreakdownPlot']:
            acqTime = datetime.strptime(node.attributes['TIME-STAMP'], '%a %b %d %H:%M:%S %Y')
            cast = f"{type(self).__name__}_{acqTime.strftime('%Y%m%d%H%M%S')}"

            p_unc = UncertaintyGUI()
            try:
                p_unc.pie_plot_class_l2(
                    rrs_means,
                    lw_means,
                    rrs_uncertainties,
                    lw_uncertainties,
                    np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str], dtype=float),  # pass radcal wavelengths
                    cast,
                    node.getGroup("ANCILLARY")
                )
                p_unc.plot_class_L2(
                    rrs_means,
                    lw_means,
                    rrs_uncertainties,
                    lw_uncertainties,
                    np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str], dtype=float),
                    cast
                )
            except ValueError as err:
                msg = f"unable to run uncertainty breakdown plots for {cast}, with error: {err}"
                print(msg)
                Utilities.writeLogFile(msg)

        # these are absolute values!
        output = {}
        rhoUNC_CWB = self.interp_common_wvls(
            rhoUNC,
            np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str], dtype=float)[ind_rad_wvl],
            waveSubset,
            return_as_dict=False
        )
        lwAbsUnc[nan_mask] = np.nan
        lwAbsUnc = self.interp_common_wvls(
            lwAbsUnc,
            np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str], dtype=float)[ind_rad_wvl],
            waveSubset,
            return_as_dict=False
        )
        rrsAbsUnc[nan_mask] = np.nan
        rrsAbsUnc = self.interp_common_wvls(
            rrsAbsUnc,
            np.array(uncGrp.getDataset(rad_cal_str).columns[cal_col_str], dtype=float)[ind_rad_wvl],
            waveSubset,
            return_as_dict=False
        )
        #print("rrsAbsUnc")
        #print(rrsAbsUnc)

        ## Band Convolution of Uncertainties
        # get unc values at common wavebands (from ProcessL2) and convert any NaNs to 0 to not create issues with punpy
        esUNC_band = np.array([i[0] for i in xSlice['esUnc'].values()])
        liUNC_band = np.array([i[0] for i in xSlice['liUnc'].values()])
        ltUNC_band = np.array([i[0] for i in xSlice['ltUnc'].values()])

        # Prune the uncertainties to remove NaNs and negative values (uncertainties which make no physical sense)
        esUNC_band[np.isnan(esUNC_band)] = 0.0
        liUNC_band[np.isnan(liUNC_band)] = 0.0
        ltUNC_band[np.isnan(ltUNC_band)] = 0.0
        esUNC_band = np.abs(esUNC_band)  # uncertainties may have negative values after conversion to relative units
        liUNC_band = np.abs(liUNC_band)
        ltUNC_band = np.abs(ltUNC_band)

        ## Update the output dictionary with band L2 hyperspectral and satellite band uncertainties
        for s_key in self._SATELLITES.keys():
            output.update(
                self.get_band_outputs(
                    s_key, rho, lw_means, lw_uncertainties, rrs_means, rrs_uncertainties,
                    esUNC_band, liUNC_band, ltUNC_band, rhoUNC, waveSubset, xSlice
                )
            )
        output.update(
            {"rhoUNC_HYPER": {str(k): val for k, val in zip(waveSubset, rhoUNC_CWB)},
            "lwUNC": lwAbsUnc,
             "rrsUNC": rrsAbsUnc}
        )

        return output

    ## Utilities ##

    @staticmethod
    def extract_unc_from_grp(grp: HDFGroup, name: str, col_name: Optional[str] = None) -> Union[np.array, HDFDataset]:
        """
        small function to avoid repetition of code

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

    @staticmethod
    def extract_factory_cal(node, radcal, s, cCal, cCoef):
        """
        small function to get the calibration and calibration uncertainty, mutates cCal and cCoef in lieu of return value
        :param node: HDF root - full HDF file
        :param radcal: HDF group containing radiometric calibration
        :param s: dict key to append data to cCal and cCoef
        :param cCal: dict for storing calibration
        :param cCoef: dict for storing calibration coeficients 
        """
         
        # ADERU : Read radiometric coeff value from configuration files
        cCal[s] = np.asarray(list(radcal.columns['unc']))
        calFolder = os.path.splitext(ConfigFile.filename)[0] + "_Calibration"
        calPath = os.path.join(PATH_TO_CONFIG, calFolder)
        calibrationMap = CalibrationFileReader.read(calPath)

        if ConfigFile.settings['SensorType'].lower() == "dalec":
            waves, cCoef[s] = ProcessL1b_FactoryCal.extract_calibration_coeff_dalec(calibrationMap, s)
        else:    
            waves, cCoef[s] = ProcessL1b_FactoryCal.extract_calibration_coeff(node, calibrationMap, s)

    def get_band_outputs(self, sensor_key: str, rho, lw_means, lw_uncertainties, rrs_means, rrs_uncertainties,
                         esUNC, liUNC, ltUNC, rhoUNC, waveSubset, xSlice) -> dict:

        """
        runs band convolution for class-based regime

        :param sensor_key: sensor key for self._SATELLITES depending on target for band conv
        :param rho: rho values provided by M99 or Z17 (depending on user settings)
        :param lw_means: class based regime mean values for Lw inputs
        :param lw_uncertainties: class based regime uncertainty values for Lw inputs
        :param rrs_means: class based regime mean values for Rrs inputs
        :param rrs_uncertainties: class based regime uncertainty values for Rrs inputs
        :param esUNC: Es uncertainty values
        :param liUNC: Li uncertainty values
        :param ltUNC: Lt uncertainty values
        :param rhoUNC: rho uncertainty values
        :param waveSubset: subset of wavelengths for L2 products to be interpolated to
        :param xSlice: dictionary for storing outputs
        """

        if ConfigFile.settings[self._SATELLITES[sensor_key]['config']]:
            sensor_name = self._SATELLITES[sensor_key]['name']
            RSR_Bands = self._SATELLITES[sensor_key]['Weight_RSR']
            prop_Band_CB = Propagate(M=100, cores=1)  # propagate band convolved uncertainties class based
            Band_Convolved_UNC = {}
            esDeltaBand = prop_Band_CB.band_Conv_Uncertainty(
                [np.asarray(list(xSlice['es'].values()), dtype=float).flatten(), waveSubset],
                [esUNC, None],
                sensor_key  # used to choose correct band convolution measurement function in uncertainty_analysis.py
            )
            # band_Conv_Uncertainty uses def_sensor_mfunc(sensor_key) to select the correct measurement function
            # per satellite. sensor_key refers to the keys of the _SATELLITE dict which were chosen to match the keys
            # in def_sensor_mfunc.

            Band_Convolved_UNC[f"esUNC_{sensor_name}"] = {
                str(k): [val] for k, val in zip(RSR_Bands, esDeltaBand)
            }

            liDeltaBand = prop_Band_CB.band_Conv_Uncertainty(
                [np.asarray(list(xSlice['li'].values()), dtype=float).flatten(), waveSubset],
                [liUNC, None],
                sensor_key
            )
            Band_Convolved_UNC[f"liUNC_{sensor_name}"] = {
                str(k): [val] for k, val in zip(RSR_Bands, liDeltaBand)
            }

            ltDeltaBand = prop_Band_CB.band_Conv_Uncertainty(
                [np.asarray(list(xSlice['lt'].values()), dtype=float).flatten(), waveSubset],
                [ltUNC, None],
                sensor_key
            )
            Band_Convolved_UNC[f"ltUNC_{sensor_name}"] = {
                str(k): [val] for k, val in zip(RSR_Bands, ltDeltaBand)
            }

            rhoDeltaBand = prop_Band_CB.band_Conv_Uncertainty(
                [rho, waveSubset],
                [rhoUNC, None],
                sensor_key
            )
            Band_Convolved_UNC[f"rhoUNC_{sensor_name}"] = {
                str(k): [val] for k, val in zip(RSR_Bands, rhoDeltaBand)
            }

            Band_Convolved_UNC[f"lwUNC_{sensor_name}"] = prop_Band_CB.Propagate_Lw_Convolved(
                lw_means,
                lw_uncertainties,
                sensor_key,
                waveSubset
            )
            Band_Convolved_UNC[f"rrsUNC_{sensor_name}"] = prop_Band_CB.Propagate_RRS_Convolved(
                rrs_means,
                rrs_uncertainties,
                sensor_key,
                waveSubset
            )

            return Band_Convolved_UNC
        else:
            return {}

    def get_band_outputs_FRM(self, sensor_key, MCP_obj, esSample, liSample, ltSample, rhoSample, sample_wavelengths
                             ) -> dict:
        """
        runs band convolution for FRM regime
        
        :param sensor_key: sensor key for self._SATELLITES depending on target for band conv
        :param MCP_obj: Monte Carlo propagation object for accessing measurment functions and punpy/comet_maths methods
        :param esSample: Monte Carlo sample (wavelengths x Mdraws) generated for Es 
        :param liSample: Monte Carlo sample (wavelengths x Mdraws) generated for Li 
        :param ltSample: Monte Carlo sample (wavelengths x Mdraws) generated for Lt 
        :param rhoSample: Monte Carlo sample (wavelengths x Mdraws) generated for rho
         """
        
        # now requires MCP_obj to be a Propagate object as def_sensor_mfunc is not a static method
        if ConfigFile.settings[self._SATELLITES[sensor_key]['config']]:
            sensor_name = self._SATELLITES[sensor_key]['name']
            RSR_Bands = self._SATELLITES[sensor_key]['Weight_RSR']
            Band_Convolved_UNC = {}

            sample_es_conv = MCP_obj.MCP.run_samples(MCP_obj.def_sensor_mfunc(sensor_key), [esSample, sample_wavelengths])
            sample_li_conv = MCP_obj.MCP.run_samples(MCP_obj.def_sensor_mfunc(sensor_key), [liSample, sample_wavelengths])
            sample_lt_conv = MCP_obj.MCP.run_samples(MCP_obj.def_sensor_mfunc(sensor_key), [ltSample, sample_wavelengths])

            sample_rho_conv = MCP_obj.MCP.run_samples(MCP_obj.def_sensor_mfunc(sensor_key), [rhoSample, sample_wavelengths])

            esDeltaBand = MCP_obj.MCP.process_samples(None, sample_es_conv)
            liDeltaBand = MCP_obj.MCP.process_samples(None, sample_li_conv)
            ltDeltaBand = MCP_obj.MCP.process_samples(None, sample_lt_conv)

            rhoDeltaBand = MCP_obj.MCP.process_samples(None, sample_rho_conv)

            sample_lw_conv = MCP_obj.MCP.run_samples(MCP_obj.Lw_FRM, [sample_lt_conv, sample_rho_conv, sample_li_conv])
            sample_rrs_conv = MCP_obj.MCP.run_samples(MCP_obj.Rrs_FRM, [sample_lt_conv, sample_rho_conv, sample_li_conv, sample_es_conv])

            # put in expected format (converted from punpy conpatible outputs) and put in output dictionary which will
            # be returned to ProcessingL2 and used to update xSlice/xUNC
            Band_Convolved_UNC[f"esUNC_{sensor_name}"] = {str(k): [val] for k, val in zip(RSR_Bands, esDeltaBand)}
            Band_Convolved_UNC[f"liUNC_{sensor_name}"] = {str(k): [val] for k, val in zip(RSR_Bands, liDeltaBand)}
            Band_Convolved_UNC[f"ltUNC_{sensor_name}"] = {str(k): [val] for k, val in zip(RSR_Bands, ltDeltaBand)}
            Band_Convolved_UNC[f"rhoUNC_{sensor_name}"] = {str(k): [val] for k, val in zip(RSR_Bands, rhoDeltaBand)}
            # L2 uncertainty products can be reported as np arrays
            Band_Convolved_UNC[f"lwUNC_{sensor_name}"] = MCP_obj.MCP.process_samples(None, sample_lw_conv)
            Band_Convolved_UNC[f"rrsUNC_{sensor_name}"] = MCP_obj.MCP.process_samples(None, sample_rrs_conv)

            return Band_Convolved_UNC
        else:
            return {}

    @staticmethod
    def apply_NaN_Mask(rawSlice):
        for wvl in rawSlice:  # iterate over wavelengths
            if any(np.isnan(rawSlice[wvl])):  # if we encounter any NaN's
                for msk in np.where(np.isnan(rawSlice[wvl]))[0]:  # mask may be multiple indexes
                    for wl in rawSlice:  # strip the scan
                        rawSlice[wl].pop(msk)  # remove the scan if nans are found anywhere

    @staticmethod
    def interp_common_wvls(columns, waves, newWaveBands, return_as_dict: bool =False) -> Union[np.array, OrderedDict]:
        """
        interpolate array to common wavebands

        :param columns: values to be interpolated (y)
        :param waves: current wavelengths (x)
        :param newWaveBands: wavelenghts to interpolate (new_x)
        :param return_as_dict: boolean which if true will return an ordered dictionary (wavelengths are keys)

        :return: returns the interpolated output as either a numpy array or Ordered-Dictionary
        """
        saveTimetag2 = None
        if isinstance(columns, dict):
            if "Datetag" in columns:
                saveDatetag = columns.pop("Datetag")
                saveTimetag2 = columns.pop("Timetag2")
                columns.pop("Datetime")
            y = np.asarray(list(columns.values()))
        elif isinstance(columns, np.ndarray):  # is numpy array
            y = columns
        else:
            msg = "columns are unexpected type: ProcessInstrumentUncertainties.py - interp_common_wvls"
            print(msg)
        # Get wavelength values
        x = np.asarray(waves)

        newColumns = OrderedDict()
        if saveTimetag2 is not None:
            newColumns["Datetag"] = saveDatetag
            newColumns["Timetag2"] = saveTimetag2
        # Can leave Datetime off at this point

        for i in range(newWaveBands.shape[0]):
            newColumns[str(round(10*newWaveBands[i])/10)] = []  # limit to one decimal place

        new_y = np.interp(newWaveBands, x, y)  #InterpolatedUnivariateSpline(x, y, k=3)(newWavebands)

        for waveIndex in range(newWaveBands.shape[0]):
            newColumns[str(round(10*newWaveBands[waveIndex])/10)].append(new_y[waveIndex])

        if return_as_dict:
            return newColumns
        else:
            return new_y

    @staticmethod
    def interpolateSamples(Columns, waves, newWavebands):
        '''
        Wavelength Interpolation for differently sized arrays containing samples
        Use a common waveband set determined by the maximum lowest wavelength
        of all sensors, the minimum highest wavelength, and the interval
        set in the Configuration Window.
        '''

        # Copy dataset to dictionary
        columns = {k: Columns[:, i] for i, k in enumerate(waves)}
        cols = []
        for m in range(Columns.shape[0]):  # across all the monte carlo draws
            newColumns = {}

            for i in range(newWavebands.shape[0]):
                # limit to one decimal place
                newColumns[str(round(10*newWavebands[i])/10)] = []

            # for m in range(Columns.shape[0]):
            # Perform interpolation for each timestamp
            y = np.asarray([columns[k][m] for k in columns])

            new_y = sp.interpolate.InterpolatedUnivariateSpline(waves, y, k=3)(newWavebands)

            for waveIndex in range(newWavebands.shape[0]):
                newColumns[str(round(10*newWavebands[waveIndex])/10)].append(new_y[waveIndex])

            cols.append(newColumns)

        return np.asarray(cols)

    def gen_n_IB_sample(self, mDraws):
        # make your own sample here min is 3, max is 6 - all values must be integer
        import random as rand
        # seed random number generator with current systime (default behaviour of rand.seed)
        rand.seed(a=None, version=2)
        sample_n_IB = []
        for i in range(mDraws):
            sample_n_IB.append(rand.randrange(3, 7, 1))  # sample_n_IB max should be 6
        return np.asarray(sample_n_IB)  # make numpy array to be compatible with comet maths

    def get_Slaper_Sl_unc(self, data, sample_data, mZ, sample_mZ, n_iter, sample_n_iter, MC_prop, mDraws):
        """
        finds the uncertainty in the slaper correction. Error estimated from the difference between slaper correction
        using n_iter and n_iter - 1

        :param data: signal to be corrected (either S12 or signal)
        :param sample_data: MC sample [PDF] of data attribute
        :param mZ: LSF read from tartu files
        :param sample mZ: PDF of mZ
        :param n_iter: number of iterations
        :param sample_n_iter: simple PDF of n_iter, no uncertainty should be passed here.
        :param MC_prop: punpy.MCP object as namespace for calling punpy functions/settings
        :param mDraws: number of monte carlo draws, M
        """
        # calculates difference between n=4 and n=5, then propagates as an error
        sl_corr = self.Slaper_SL_correction(data, mZ, n_iter)
        sl_corr_unc = []
        sl4 = self.Slaper_SL_correction(data, mZ, n_iter=n_iter - 1)
        for i in range(len(sl_corr)):  # get the difference between n=4 and n=5
            if sl_corr[i] > sl4[i]:
                sl_corr_unc.append(sl_corr[i] - sl4[i])
            else:
                sl_corr_unc.append(sl4[i] - sl_corr[i])

        sample_sl_syst = cm.generate_sample(mDraws, sl_corr, np.array(sl_corr_unc), "syst")
        sample_sl_rand = MC_prop.run_samples(self.Slaper_SL_correction, [sample_data, sample_mZ, sample_n_iter])
        sample_sl_corr = MC_prop.combine_samples([sample_sl_syst, sample_sl_rand])

        return sample_sl_corr

    # Measurement Functions
    @staticmethod
    def S12func(k, S1, S2):
        "compares DN at two separate times, part of non linearity correction derrivation"
        return ((1 + k)*S1) - (k*S2)

    @staticmethod
    def alphafunc(S1, S12):
        t1 = [Decimal(S1[i]) - Decimal(S12[i]) for i in range(len(S1))]
        t2 = [pow(Decimal(S12[i]), 2) for i in range(len(S12))]
        # I added a conditional to check if any values in S12 are zero. One value of S12 was 0 which caused issue #253
        return np.asarray([float(t1[i]/t2[i]) if t2[i] != 0 else 0 for i in range(len(t1))])

    @staticmethod
    def dark_Substitution(light, dark):
        return light - dark

    @staticmethod
    def non_linearity_corr(offset_corrected_mesure, alpha):
        linear_corr_mesure = offset_corrected_mesure*(1 - alpha*offset_corrected_mesure)
        return linear_corr_mesure

    @staticmethod
    def Zong_SL_correction(input_data, C_matrix):
        return np.matmul(C_matrix, input_data)

    @staticmethod
    def Slaper_SL_correction(input_data, SL_matrix, n_iter=5):
        nband = len(input_data)
        m_norm = np.zeros(nband)

        mC = np.zeros((n_iter + 1, nband))
        mX = np.zeros((n_iter + 1, nband))
        mZ = SL_matrix
        mX[0, :] = input_data

        for i in range(nband):
            jstart = np.max([0, i - 10])
            jstop = np.min([nband, i + 10])
            m_norm[i] = np.sum(mZ[i, jstart:jstop])  # eq 4

        for i in range(nband):
            if m_norm[i] == 0:
                mZ[i, :] = np.zeros(nband)
            else:
                mZ[i, :] = mZ[i, :]/m_norm[i]  # eq 5

        for k in range(1, n_iter + 1):
            for i in range(nband):
                mC[k - 1, i] = mC[k - 1, i] + np.sum(mX[k - 1, :]*mZ[i, :])  # eq 6
                if mC[k - 1, i] == 0:
                    mX[k, i] = 0
                else:
                    mX[k, i] = (mX[k - 1, i]*mX[0, i])/mC[k - 1, i]  # eq 7

        return mX[n_iter - 1, :]

    @staticmethod
    def absolute_calibration(normalized_mesure, updated_radcal_gain):
        return normalized_mesure/updated_radcal_gain

    @staticmethod
    def thermal_corr(Ct, calibrated_mesure):
        return Ct*calibrated_mesure

    @staticmethod
    def prepare_cos(uncGrp, sensortype, level=None, ind_raw_wvl=None):
        """
        read from hdf and prepare inputs for cos_err measurement function
        """
        ## Angular cosine correction (for Irradiance)
        if level != 'L2':
            radcal_wvl = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['1'][1:].tolist())
            coserror = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").data))[1:, 2:]
            cos_unc = (np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY").data))[1:, 2:]
                       /100)*np.abs(coserror)

            coserror_90 = np.asarray(
                pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR_AZ90").data))[1:, 2:]
            cos90_unc = (np.asarray(
                pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY_AZ90").data))[1:,
                         2:]/100)*np.abs(coserror_90)
        else:
            # reading in data changes if at L2 (because hdf files have different layout)
            radcal_wvl = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['1'][1:].tolist())
            coserror = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").data))[1:, 2:]
            cos_unc = (np.asarray(
                pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY").data))[1:, 2:]/100)*np.abs(coserror)
            coserror_90 = np.asarray(
                pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR_AZ90").data))[1:, 2:]
            cos90_unc = (np.asarray(
                pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY_AZ90").data))[1:, 2:]/100)*np.abs(coserror_90)

        radcal_unc = None  # no uncertainty in the wavelengths as they are only used to index

        zenith_ang = uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").attributes["COLUMN_NAMES"].split('\t')[2:]
        zenith_ang = np.asarray([float(x) for x in zenith_ang])
        zen_unc = np.asarray([0.05 for x in zenith_ang])  # default of 0.5 for solar zenith unc

        if ind_raw_wvl is not None:
            radcal_wvl = radcal_wvl[ind_raw_wvl]
            coserror = coserror[ind_raw_wvl]
            coserror_90 = coserror_90[ind_raw_wvl]
            cos_unc = cos_unc[ind_raw_wvl]
            cos90_unc = cos90_unc[ind_raw_wvl]

        return [radcal_wvl, coserror, coserror_90, zenith_ang], [radcal_unc, cos_unc, cos90_unc, zen_unc]

    @staticmethod
    def AZAvg_Coserr(coserror, coserror_90):
        # if delta < 2% : averaging the 2 azimuth plan
        return (coserror + coserror_90)/2.  # average azi coserr

    @staticmethod
    def ZENAvg_Coserr(radcal_wvl, AZI_avg_coserror):
        i1 = np.argmin(np.abs(radcal_wvl - 300))
        i2 = np.argmin(np.abs(radcal_wvl - 1000))

        # if delta < 2% : averaging symetric zenith
        ZEN_avg_coserror = (AZI_avg_coserror + AZI_avg_coserror[:, ::-1])/2.

        # set coserror to 1 outside range [450,700]
        ZEN_avg_coserror[0:i1, :] = 0
        ZEN_avg_coserror[i2:, :] = 0
        return ZEN_avg_coserror

    @staticmethod
    def FHemi_Coserr(ZEN_avg_coserror, zenith_ang):
        # Compute full hemisperical coserror
        zen0 = np.argmin(np.abs(zenith_ang))
        zen90 = np.argmin(np.abs(zenith_ang - 90))
        deltaZen = (zenith_ang[1::] - zenith_ang[:-1])

        full_hemi_coserror = np.zeros(ZEN_avg_coserror.shape[0])

        for i in range(ZEN_avg_coserror.shape[0]):
            full_hemi_coserror[i] = np.sum(
                ZEN_avg_coserror[i, zen0:zen90]*np.sin(2*np.pi*zenith_ang[zen0:zen90]/180)*deltaZen[
                                                                                           zen0:zen90]*np.pi/180)

        return full_hemi_coserror

    @staticmethod
    def cosine_corr(avg_coserror, full_hemi_coserror, zenith_ang, thermal_corr_mesure, sol_zen, dir_rat):
        ind_closest_zen = np.argmin(np.abs(zenith_ang - sol_zen))
        cos_corr = 1 - avg_coserror[:, ind_closest_zen]/100
        Fhcorr = 1 - np.array(full_hemi_coserror)/100
        cos_corr_mesure = (dir_rat*thermal_corr_mesure*cos_corr) + ((1 - dir_rat)*thermal_corr_mesure*Fhcorr)

        return cos_corr_mesure

    @staticmethod
    def get_cos_corr(zenith_angle, solar_zenith, cosine_error):
        ind_closest_zen = np.argmin(np.abs(zenith_angle - solar_zenith))
        return 1 - cosine_error[:, ind_closest_zen]/100

    @staticmethod
    def cos_corr(signal, direct_ratio, cos_correction, full_hemi_cos_error):
        Fhcorr = (1 - full_hemi_cos_error / 100)
        return (direct_ratio * signal * cos_correction) + ((1 - direct_ratio) * signal * Fhcorr)

    @staticmethod
    def cos_corr_fun(avg_coserror, zenith_ang, sol_zen):
        ind_closest_zen = np.argmin(np.abs(zenith_ang - sol_zen))
        return 1 - avg_coserror[:, ind_closest_zen]/100

    @staticmethod
    def cosine_error_correction(uncGrp, sensortype):

        ## Angular cosine correction (for Irradiance)
        radcal_wvl = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_RADCAL_CAL").data)['1'][1:].tolist())
        coserror = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").data))[1:,2:]
        coserror_90 = np.asarray(pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR_AZ90").data))[1:, 2:]
        coserror_unc = (np.asarray(
            pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY").data))[1:,2:]/100)*coserror
        coserror_90_unc = (np.asarray(
            pd.DataFrame(uncGrp.getDataset(sensortype + "_ANGDATA_UNCERTAINTY_AZ90").data))[1:, 2:]/100)*coserror_90
        zenith_ang = uncGrp.getDataset(sensortype + "_ANGDATA_COSERROR").attributes["COLUMN_NAMES"].split('\t')[2:]
        i1 = np.argmin(np.abs(radcal_wvl - 300))
        i2 = np.argmin(np.abs(radcal_wvl - 1000))
        zenith_ang = np.asarray([float(x) for x in zenith_ang])

        # comparing cos_error for 2 azimuth
        AZI_delta_err = np.abs(coserror - coserror_90)

        # if delta < 2% : averaging the 2 azimuth plan
        AZI_avg_coserror = (coserror + coserror_90)/2.
        AZI_delta = np.power(np.power(coserror_unc, 2) + np.power(coserror_90_unc, 2), 0.5)  # TODO: check this!

        # comparing cos_error for symetric zenith
        ZEN_delta_err = np.abs(AZI_avg_coserror - AZI_avg_coserror[:, ::-1])
        ZEN_delta = np.power(np.power(AZI_delta, 2) + np.power(AZI_delta[:, ::-1], 2), 0.5)

        # if delta < 2% : averaging symetric zenith
        ZEN_avg_coserror = (AZI_avg_coserror + AZI_avg_coserror[:, ::-1])/2.

        # set coserror to 1 outside range [450,700]
        ZEN_avg_coserror[0:i1, :] = 0
        ZEN_avg_coserror[i2:, :] = 0

        return ZEN_avg_coserror, AZI_avg_coserror, zenith_ang, ZEN_delta_err, ZEN_delta, AZI_delta_err, AZI_delta


    @staticmethod
    def read_sixS_model(node):
        res_sixS = {}
        
        # Create a temporary group to pop date time columns
        newGrp = node.addGroup('temp')
        newGrp.copy(node.getGroup('SIXS_MODEL'))
        for ds in newGrp.datasets:
            newGrp.datasets[ds].datasetToColumns()
        sixS_gp = node.getGroup('temp')
        
        sixS_gp.getDataset("direct_ratio").columns.pop('Datetime')
        sixS_gp.getDataset("direct_ratio").columns.pop('Timetag2')
        sixS_gp.getDataset("direct_ratio").columns.pop('Datetag')
        sixS_gp.getDataset("direct_ratio").columnsToDataset()
        sixS_gp.getDataset("diffuse_ratio").columns.pop('Datetime')
        sixS_gp.getDataset("diffuse_ratio").columns.pop('Timetag2')
        sixS_gp.getDataset("diffuse_ratio").columns.pop('Datetag')
        sixS_gp.getDataset("diffuse_ratio").columnsToDataset()

        # sixS_gp.getDataset("direct_ratio").datasetToColumns()
        res_sixS['solar_zenith'] = np.asarray(sixS_gp.getDataset('solar_zenith').columns['solar_zenith'])
        res_sixS['wavelengths'] = np.asarray(list(sixS_gp.getDataset('direct_ratio').columns.keys())[2:], dtype=float)
        if 'timetag' in res_sixS['wavelengths']:
            # because timetag2 was included for some data and caused a bug
            res_sixS['wavelengths'] = res_sixS['wavelengths'][1:]
        res_sixS['direct_ratio'] = np.asarray(pd.DataFrame(sixS_gp.getDataset("direct_ratio").data))
        res_sixS['diffuse_ratio'] = np.asarray(pd.DataFrame(sixS_gp.getDataset("diffuse_ratio").data))
        node.removeGroup(sixS_gp)
        return res_sixS