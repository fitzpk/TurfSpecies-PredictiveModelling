# Web Scraping Script used to Extract Golf Course Data
# for USA tracks
# December 2018
# Kevin Fitzgerald

import requests
import numpy as np
import pandas as pd
import re
from datetime import datetime

startTime = datetime.now()
#**************************************************

# Get HTML content from Golf Course Directory Website
url = "http://www.golfnationwide.com/US-Golf-Course-List-And-Directory.aspx"
webpage = requests.get(url)
webpage = webpage.text

# Strip down the HTML so its just the section with US State links 
indexstart = webpage.index('Browse U.S. Golf Courses by State')
indexend = webpage.index('<div id="Footer">')
webpage = webpage[indexstart:indexend]

#Split HTML content by </td> tag
webpage = webpage.split('</td>')
del webpage[-1]

# Loop through each item in webpage
states=[]
for i in webpage:
    # Strip content down to just the link using indices of certain elements
    link = i[i.index('href'):i.index('aspx')+4]
    link = link.replace('href="','')
    # Generate the link for each state and append it to a list
    link = 'http://www.golfnationwide.com' + link
    states.append(link)    

courses=[]
for i in states:
    webpage = requests.get(i)
    webpage = webpage.text
    indexstart = webpage.index('<th scope="col">')
    indexend = webpage.index('<div id="Footer">')
    webpage = webpage[indexstart:indexend]
    webpage = webpage.split('</a>')
    del webpage[-1]
    for i in webpage:
        link = i[i.index('href'):i.index('aspx')+4]
        link = link.replace('href="','')
        link = 'http://www.golfnationwide.com' + link
        courses.append(link)

coursenames=[]
courseadds=[]
coursecities=[]
coursestates=[]
coursezips=[]
coursecountry=[]

greens=[]
fairways=[]
waterHazard=[]
spikesAllowed=[]
privacy=[]
yearEst=[]
designers=[]
seasonUsage=[]

champteeslopes=[]
champteeratings=[]

for i in courses:
    webpage = requests.get(i)
    webpage = webpage.text
    try:
        info = webpage[webpage.index('<div id="BasicInfo">'):webpage.index('<div id="DetailedBox">')]
        coursename = info[info.index('CourseNameLabel'):info.index('</span>')]
        coursename = coursename.replace('CourseNameLabel">','')
        coursenames.append(coursename)
        
        courseadd = info[info.index('AddressLabel'):]
        courseadd = courseadd[:courseadd.index('</span>')]
        courseadd = courseadd.replace('AddressLabel">','')
        courseadds.append(courseadd)
     
        coursecity = info[info.index('CityLabel'):]
        coursecity = coursecity[:coursecity.index('</span>')]
        coursecity = coursecity.replace('CityLabel">','')
        coursecities.append(coursecity)

        coursestate = info[info.index('StateLabel'):]
        coursestate = coursestate[:coursestate.index('</span>')]
        coursestate = coursestate.replace('StateLabel">','')    
        coursestates.append(coursestate)
        
        coursezip = info[info.index('ZipLabel'):]
        coursezip = coursezip[:coursezip.index('</span>')]
        coursezip = coursezip.replace('ZipLabel">','')
        coursezips.append(coursezip)
        
        coursecountry.append('United States of America')

        #GET COURSE DETAILS
        courseInfo = webpage[webpage.index('GreensBox'):webpage.index('<table')]

        greentype = courseInfo[courseInfo.index('GreensLabel'):]
        greentype = greentype[:greentype.index('</span>')]
        greentype = greentype.replace('GreensLabel">','')
        greens.append(greentype)

        fairwaytype = courseInfo[courseInfo.index('FairwaysLabel'):]
        fairwaytype = fairwaytype[:fairwaytype.index('</span>')]
        fairwaytype = fairwaytype.replace('FairwaysLabel">','')
        fairways.append(fairwaytype)

        waterHaz = courseInfo[courseInfo.index('WaterHazardsLabel'):]
        waterHaz = waterHaz[:waterHaz.index('</span>')]
        waterHaz = waterHaz.replace('WaterHazardsLabel">','')
        waterHazard.append(waterHaz)

        spikes= courseInfo[courseInfo.index('MetalSpikesLabel'):]
        spikes = spikes[:spikes.index('</span>')]
        spikes = spikes.replace('MetalSpikesLabel">','')
        spikesAllowed.append(spikes)

        clubInfo = webpage[webpage.index('DetailedBox'):webpage.index('BlockTwo')]

        privacytype = clubInfo[clubInfo.index('ClassificationLabel'):]
        privacytype = privacytype[:privacytype.index('</span>')]
        privacytype = privacytype.replace('ClassificationLabel">','')
        privacy.append(privacytype)

        built = clubInfo[clubInfo.index('YearBuiltLabel'):]
        built = built[:built.index('</span>')]
        built = built.replace('YearBuiltLabel">','')
        yearEst.append(built)

        designer = clubInfo[clubInfo.index('DesignerLabel'):]
        designer = designer[:designer.index('</span>')]
        designer = designer.replace('DesignerLabel">','')
        designers.append(designer)

        usage = clubInfo[clubInfo.index('SeasonLabel'):]
        usage = usage[:usage.index('</span>')]
        usage = usage.replace('SeasonLabel">','')
        seasonUsage.append(usage)

        #GET RATINGS/SLOPE DATA
        coursetable = webpage[webpage.index('<table'):webpage.index('<div id="RatingsBox">')]
        champteeslope = coursetable[coursetable.index('ChampionshipTeeSlopeLabel">'):]
        champteeslope = champteeslope[:champteeslope.index('</span>')]
        champteeslope = champteeslope.replace('ChampionshipTeeSlopeLabel">','')
        champteeslopes.append(champteeslope)

        champteerating = coursetable[coursetable.index('ChampionshipTeeUSGARatingLabel">'):]
        champteerating = champteerating[:champteerating.index('</span>')]
        champteerating = champteerating.replace('ChampionshipTeeUSGARatingLabel">','')
        champteeratings.append(champteerating)
        
    except ValueError:
        pass

usacourses = pd.DataFrame(
    {'Name': coursenames,
     'City': coursecities,
     'Address': courseadds,
     'State': coursestates,
     'Zip Code': coursezips,
     'Country': coursecountry,
     'Green Type': greens,
     'Fairway Type': fairways,
     'Water Hazards': waterHazard,
     'Spikes Allowed': spikesAllowed,
     'Privacy': privacy,
     'Year Established': yearEst,
     'Designer': designers,
     'Season Availability': seasonUsage,
     'Slope': champteeslopes,
     'Rating': champteeratings
    }
)

usacourses.to_csv('/Users/kf/Desktop/usacourses2.csv',index=False,encoding='iso-8859-1')
           
#*********************************
print(datetime.now() - startTime)
