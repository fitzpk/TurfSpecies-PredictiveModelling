import numpy as np
import pandas as pd
import re

climates = pd.read_csv("/Users/kf/Desktop/usacityclimate.csv",encoding='iso-8859-1')
print("\nCount of Null Values per Column:\n",climates.isnull().sum())

AnnualHigh=[]
AnnualLow=[]
AverageTemp=[]
AvgAnnualRain=[]

for i in climates:
    if i not in ['City','State']:
        print(i)
        for row in climates[i]:
            lasttag = row.rfind("</")
            field = row[:lasttag]
            lasttag = field.rfind(">")+1
            row = field[lasttag:]
            if i in ['0']:
                row = row.replace('ÁF','')
                AnnualHigh.append(row)
            elif i in ['1']:
                row = row.replace('ÁF','')
                AnnualLow.append(row)
            elif i in ['2']:
                row = row.replace('ÁF','')
                AverageTemp.append(row)
            elif i in ['3']:
                row = row.replace('inch','')
                AvgAnnualRain.append(row)
            else:
                pass

#Find the mean Average Annual Rain
means=[]
for row in AvgAnnualRain:
    if row == '-':
        pass
    else:
        row = float(row)
        means.append(row)
mean = sum(means)/len(means)

#Replace the '-' values with the mean of the column
AvgAnnualRain2=[]
for row in AvgAnnualRain:
    if row == '-':
        row = str(mean)
    AvgAnnualRain2.append(row)

cities=[]
for row in climates['City']:
    row = row.replace(' ','')
    row = row.replace('-',' ')
    row = row.lower()
    cities.append(row)

newclimates = pd.DataFrame(
    {'AnnualHigh(F)': AnnualHigh,
     'AnnualLow(F)': AnnualLow,
     'AverageTemp(F)': AverageTemp,
     'AverageAnnualRain(In)': AvgAnnualRain2,
     'City': cities,
     'State': climates['State'],
    }
)

newclimates = newclimates[(newclimates['AnnualHigh(F)'] != '0') & (newclimates['AnnualLow(F)'] != '0') & (newclimates['AverageTemp(F)'] != '0')]          

newclimates.to_csv('/Users/kf/Desktop/cityclimates.csv',index=False)
