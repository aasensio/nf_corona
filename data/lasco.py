import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

timerange = a.Time('2009/03/15 21:00', '2009/03/29 15:00')
instrument = a.Instrument.lasco
detector = a.Detector.c2
result = Fido.search(timerange, instrument, detector)