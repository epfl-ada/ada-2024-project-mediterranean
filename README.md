# User Taste Analysis by Demographics

Our goal is to investigate the correlation between beer preferences and user demographics, such as age, gender, and location. Specifically, we aim to explore generational trends in beer tastes and flavor preferences. However, a challenge we face is the lack of access to users' age and gender information. To adapt our analysis, we propose the following approaches:

## 1. Demographic Analysis

We will conduct a demographic study to analyze user distribution and clustering, aiming to determine whether the success of a beer is influenced by demographic factors and temporal changes. This approach allows us to circumvent the need for specific age and gender data.

## 2. Analysis Objectives

### 2.1 Users Analysis

We will examine how users rate beers by:

* Segmenting users by location to identify where the two ranking sites are most popular. We will analyze:
    * The number of reviews and ratings submitted by users on each site per country (usage information).
    * The number of users per country (popularity information).
* Extracting user preferences for each geographical location based on their ratings and reviews.
* Utilizing the collected data to track changes in the evaluation of the same beer over time. This will help us determine whether fluctuations in ratings are due to issues with the beer itself or changes in timing.
* Using rating and review information to understand beer users’ preferences (we need to process this kind of text data in some way to extract some form of rating for that)

### 2.2 Breweries Analysis

We will investigate the locations of breweries to understand distribution patterns, which will enhance our insights from the users' analysis.
### 2.3	Beers Analysis

We will correlate beer scores obtained from users to the geographical locations of breweries, providing a clearer context for interpreting the data.


## Project Structure



## Repository Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```