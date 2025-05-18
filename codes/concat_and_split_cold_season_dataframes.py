#!/home/mileshsieh/anaconda3/bin/python
import glob
import pandas as pd

m='TaiESM1'
sce='ssp585'
fList=sorted(glob.glob(f'../data/cold_season_{m}_{sce}*'))
dfList=[]
for f in fList:
    dfList.append(pd.read_csv(f))

dfall=pd.concat(dfList)
for yr in range(2021,2100,10):
    print(yr)
    df=dfall.query(f'date >= "{yr}-01-01" and date <= "{yr+9}-12-31"')
    df.to_csv(f'../data/cold_season_{m}_{sce}_{yr}0101-{yr+9}1231.indices.csv',index=False)
