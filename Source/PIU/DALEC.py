# PIU
from Source.PIU.BaseInstrument import BaseInstrument


class DALEC(BaseInstrument):
    def __init__(self):
        super().__init__()
        self.name = 'Dalec'

    def lightDarkStats(self, grp, slice, sensortype):
        pass
    