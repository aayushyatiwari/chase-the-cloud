from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def check(sample):
    '''
    explore a raw sample
    returns bt value
    '''
    ds = Dataset(f'../data/raw/{sample}.nc')

    fk1 = ds.variables['planck_fk1'][:]
    fk2 = ds.variables['planck_fk2'][:]
    bc1 = ds.variables['planck_bc1'][:]
    bc2 = ds.variables['planck_bc2'][:]

    rad = ds.variables['Rad'][:]
    BT = (fk2 / np.log(fk1 / rad + 1) - bc1) / bc2

    print('BT min/max:', BT.min(), BT.max())

    def parse_goes_time(filename):
        part = [p for p in filename.split('_') if p.startswith('s')][0]
        s = part[1:]
        year = int(s[0:4])
        doy  = int(s[4:7])
        hour = int(s[7:9])
        minute = int(s[9:11])
        second = int(s[11:13])
        dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second)
        return dt

    plt.figure(figsize=(12, 8))
    plt.imshow(BT, cmap='gray_r', vmin=200, vmax=300)
    plt.colorbar(label='Brightness Temperature (K)')
    plt.title(str(parse_goes_time(sample)))
    plt.show()
    return BT

if __name__ == "__main__":
    sample = input("give sample name (without .nc): ")
    check(sample)