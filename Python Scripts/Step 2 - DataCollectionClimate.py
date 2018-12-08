#**************************************************
# Web Scraping Script used to Extract Climate Data
# August 23rd 2018
# Kevin Fitzgerald
#**************************************************

import requests
import numpy as np
import pandas as pd
import re
from datetime import datetime

startTime = datetime.now()
print(startTime)
#**************************************************
url = "https://www.usclimatedata.com/"
webpage = requests.get(url)
webpage = webpage.text

indexstart = webpage.index('<h3>Select a state by name</h3>')
indexend = webpage.index('id="google_ad_content_right')
webpage = webpage[indexstart:indexend]

#Split HTML by </td> tag
webpage = webpage.split('</td>')
del webpage[-1]

states=[]
for i in webpage:
    link = i[i.find('href'):i.find('" title')]
    link = link.replace('href="','')
    link = 'https://www.usclimatedata.com' + link
    states.append(link)

cities=[]
fields=[]
statenames=[]
for i in states:
    webpage = requests.get(i)
    
    #Get the state name real quick
    statename = i[i.find('climate/'):i.find('/united-states')]
    statename = statename.replace("climate/","")
    statenames.append(statename)
    
    webpage = webpage.text
    indexstart = webpage.index('<h3>Select a city by name</h3>')
    indexend = webpage.index('id="google_ad_content_right')
    citydiv = webpage[indexstart:indexend]
    citydiv = citydiv.split('</td>')
    del citydiv[-1]
    
    #get the webpages for each cities in each state
    for c in citydiv:
        link = c[c.find('href'):c.find('" title')]
        link = link.replace('href="','')
        if link == '':
            pass
        else:
            link = 'https://www.usclimatedata.com' + link
            cities.append(link)
        
    #get the climate data for each state -- to be used to fill null data
    indexstart = webpage.index('id="climate_totals')
    indexend = webpage.index('id="extra_info"')
    climateTable = webpage[indexstart:indexend]
    climateFields = climateTable.split('</tr>')
    del climateFields[-1]
    for f in climateFields:
        lasttag = f.rfind("</")
        field = f[:lasttag]
        lasttag = field.rfind(">")+1
        field = field[lasttag:]
        fields.append(field)

#Create a dictionary that takes the state name and
#loops through the list of values 
cnt=7
diction={}
for i in statenames:
    startidx = cnt-7
    vals = fields[startidx:cnt]
    diction.update({i:[vals]})
    cnt+=7
#list(diction.values())[1]
stateClimate = pd.DataFrame.from_dict(diction,orient='index')

AnnualHigh=[]
AnnualLow=[]
AverageTemp=[]
AvgAnnualRain=[]
RainDaysInYear=[]
AnnualSunHours=[]
AvgAnnualSnow=[]

for i in stateClimate[0]:
    print(i)
    AnnualHigh.append(i[0])
    AnnualLow.append(i[1])
    AverageTemp.append(i[2])
    AvgAnnualRain.append(i[3])
    RainDaysInYear.append(i[4])
    AnnualSunHours.append(i[5])
    AvgAnnualSnow.append(i[6])

STclimate = pd.DataFrame(
    {'Annual High Temp': AnnualHigh,
     'Annual Low Temp': AnnualLow,
     'Average Temp': AverageTemp,
     'Average Annual Rain': AvgAnnualRain,
     'Rain Days per Year': RainDaysInYear,
     'Annual Sun Hours': AnnualSunHours,
     'Average Annual Snow Fall': AvgAnnualSnow,
     'State': statenames,
    }
)

STclimate.to_csv('/Users/kf/Desktop/usaclimate.csv',index=False,encoding='iso-8859-1')

#*******************************************************
# Retrieve the climate data for each city in each state

print('Checkpoint - ',datetime.now() - startTime)
print(len(cities))

cityfields=[]
statelocs=[]
citylocs=[]
for i in cities:
    webpage = requests.get(i)

    Instate = i[i.find('climate/'):i.find('/united-states')]
    Instate = Instate.replace("climate/","")
    Cit = Instate[:Instate.find("/")]
    Instate = Instate[Instate.find("/")+1:]
    statelocs.append(Instate)
    citylocs.append(Cit)
    
    webpage = webpage.text
    indexstart = webpage.index('id="climate_totals')
    indexend = webpage.index('id="buttons')
    climateTable = webpage[indexstart:indexend]
    climateFields = climateTable.split('</tr>')
    del climateFields[-1]
    cityfields.append(climateFields)
    
fieldsdf = pd.DataFrame(cityfields)

print(datetime.now() - startTime)
    
CYclimate = pd.DataFrame(
    {'City': citylocs,
     'State': statelocs
    }
)
CYclimate = pd.concat([CYclimate,fieldsdf],axis=1)

CYclimate.to_csv('/Users/kf/Desktop/usacityclimate2.csv',index=False,encoding='iso-8859-1')
           
#*********************************
print(datetime.now() - startTime)
