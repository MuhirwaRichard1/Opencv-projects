
import datetime as dt

class TimeStamp:

    def __init__(self, epoch_secs: int, epoch_milli_secs: int):
        self.epoch_secs = epoch_secs
        self.epoch_milli_secs = epoch_milli_secs

    # getter for epoch_secs
    @property
    def epoch_secs(self):
        return self._epoch_secs
    
    # setter for the epoch_secs
    @epoch_secs.setter
    def epoch_secs(self, value):
        if not isinstance(value, int):
            raise TypeError("Epoch seconds given in TimeStamp() class must be int only")
        self._epoch_secs = value

    # getter for epoch_milli_secs
    @property
    def epoch_milli_secs(self):
        return self._epoch_milli_secs
    
    # setter for the epoch_milli_secs
    @epoch_milli_secs.setter
    def epoch_milli_secs(self, value):
        if not isinstance(value, int):
            raise TypeError("Epoch milliseconds given in TimeStamp() class must be int only")
        if value < 0 or value >999:
            raise ValueError("Value of milliseconds in epochs must be between 0 and 999")
        self._epoch_milli_secs = value

    # getter for date_time
    @property
    def date_time(self):
        epoch = int(str(self.epoch_secs) + str(self.epoch_milli_secs).zfill(3))/1000.0
        print(epoch)
        self._date_time = dt.datetime.utcfromtimestamp(epoch)
        return self._date_time
    
    def __repr__(self):
        out = "[" + str(self.date_time) + "]" + " epoch = [" + str(self.epoch_secs) + "." + str(self.epoch_milli_secs).zfill(3) +"]\n"
        return out
    
    def __sub__(self, other):

        if not isinstance(other, TimeStamp):
            raise TypeError("Both variables must be of type TimeStamp() to calculate the time difference")

        diff = self.date_time - other.date_time
        return (diff.total_seconds())