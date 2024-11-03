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
* 
* 
* 




### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

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