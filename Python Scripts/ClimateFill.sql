DROP TABLE climate;
CREATE TABLE climate
(
  AnnualHigh numeric(3,1),
  AnnualLow numeric(3,1),
  AverageTemp numeric(3,1),
  AverageAnnualRain numeric(5,2),
  City character varying,
  State character(2)
)

DROP TABLE stateclimate;
CREATE TABLE stateclimate
(
  StateAnnualHigh numeric(3,1),
  StateAnnualLow numeric(3,1),
  StateAverageTemp numeric(3,1),
  StateAverageAnnualRain numeric(5,2),
  StateT character(2)
)

DROP TABLE courses;
CREATE TABLE courses
(
  Name character varying,
  CourseCity character varying,
  Address character varying,
  CourseState character(2),
  ZipCode character varying,
  Country character(25),
  GreenType character varying,
  FairwayType character varying,
  WaterHazard character varying,
  SpikesAllowed character varying,
  Privacy character varying,
  YearEstablished character varying,  
  Designer character varying,
  SeasonAvailability character varying,
  Slope character varying,
  Rating character varying
)

SELECT * FROM climate;
SELECT * FROM courses;

/*JOIN THE CITY CLIMATE DATA TO THE COURSE DATA BY MATCHING BOTH 
  CITY AND STATE DATA */
CREATE TABLE CourClim AS 
	(SELECT *
	FROM courses
	LEFT JOIN climate ON LOWER(climate.City) = LOWER(courses.CourseCity) 
	AND LOWER(climate.State) = LOWER(courses.CourseState));

SELECT * FROM CourClim;

/*JOIN THE STATE CLIMATE DATA TO THE COURSE DATA BY MATCHING STATE DATA -
  THIS IS DONE TO EVENTUALLY FILL IN THE UNMATCHED CITY VALUES */
CREATE TABLE CourClim2 AS 
	(SELECT *
	FROM CourClim
	LEFT JOIN stateclimate ON stateclimate.StateT = CourClim.CourseState);

/*CREATE FINAL FILLED DATASET - CASE STAEMENTS ARE USED TO USE THE STATE CLIMATE DATA
  WHEN THE CITY CLIMATE DATA IS NULL */
CREATE TABLE CourseClimate AS 
(
SELECT Name, CourseCity, Address, CourseState, ZipCode, Country, GreenType, FairwayType, 
WaterHazard, SpikesAllowed, Privacy, YearEstablished, Designer, SeasonAvailability, Slope, Rating,  
CASE
WHEN AnnualHigh IS NULL THEN StateAnnualHigh
ELSE AnnualHigh
END,
CASE
WHEN AnnualLow IS NULL THEN StateAnnualLow
ELSE AnnualLow
END,
CASE
WHEN AverageTemp IS NULL THEN StateAverageTemp
ELSE AverageTemp
END,
CASE
WHEN AverageAnnualRain IS NULL THEN StateAverageAnnualRain
ELSE AverageAnnualRain
END
FROM CourClim2
);

/*EXPORT TO CSV*/
COPY CourseClimate TO '/Users/kf/Desktop/finaltable.csv' DELIMITER ',' CSV HEADER;