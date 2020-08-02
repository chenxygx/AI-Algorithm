import json
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

path = 'bitly_usagov/example.txt'

open(path).readline()
records = [json.loads(line) for line in open(path)]
# print(records[0]['tz'])

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10])

frame = DataFrame(records)
# print(frame)

tz_counts = frame['tz'].value_counts()
# print(tz_counts)

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system[:10])
