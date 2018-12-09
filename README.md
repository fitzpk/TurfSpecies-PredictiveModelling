# Predictive Modelling Meets Turf Management
Using a Multi-Output Classifier to Predict the Turf/Grass Species of Golf Course Greens and Fairways.


#### Introduction & Purpose
From the Bermuda Grass greens found throughout Florida to the Fescue fairways of Whistling Straits Golf Course in Wisconsin, there is a vast range of turf types used for U.S. golf courses and it is always expanding as climate change, sustainability, and player preferences constantly provoke the development of new hybrid species. The purpose of this project is to create a multi-output classification model that utilizes a decision tree classifier to predict the turf species for both the greens and fairways of United States golf courses. To achieve this, Postgres and Python software will be used in conjunction with a combination of golf course data and climate data to inform the predictive analysis.

#### Data Sources
Data for this project was extracted, via Web Scraping with the requests library, from the following websites:
- http://www.golfnationwide.com/US-Golf-Course-List-And-Directory.aspx
- https://www.usclimatedata.com/

3 datasets were created during this process:
- Golf Course Data (17154 Rows, 17 Features)
- State Climate Data (51 Rows, 5 Features)
- City Climate Data (5796 rows, 6 Features)

These datasets were joined using left-join statements in Postgres. The final output was then brought into Python for transformations and model prediction. 
