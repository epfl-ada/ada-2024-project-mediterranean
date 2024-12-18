# Global Beer Trends: Demographic Analysis and Flavor Evolution Across Continents

## **Abstract**

Our tastes reveal much about who we are and the story we carry with us. That’s why we aim to present a **demographic and temporal analysis of beer preferences**. We want to investigate how consumer preferences for beer shift year by year on both a continental and global scale. Our goal is to identify the top-ranked beers worldwide and to capture their unique flavor profiles through user reviews. Ultimately, we aim to understand the long-term evolution of taste preferences and predict the defining traits of future favorites. For this study, we will use data from the datasets: BeerAdvocate and RateBeer.

By analyzing emerging trends, we seek to forecast which aromas and styles will drive choices in the near future, helping shape the global beer scene in the years to come.

> ### A journey through the world’s flavors, uncovering the beer trends that will define tomorrow!

## **Research questions**

* **How are users and breweries distributed in the datasets we are analyzing?** It's important to consider this distribution when weigthing and interpreting the results of the analysis.

* **What is the top-rated beer each year for each continent, and which beer ranks as the global favorite?**

* **How do the flavor profiles of globally preferred beers evolve over time?** Can we identify a trend toward beers with specific aromatic characteristics?

* **Is taste evolving on a global scale, or do we observe more distinct trends at the continental level?**

* **Is the most reviewed beer also the best?** Does the most popular beer also hold the title of the favorite? What carries more weight: fame or data?

* **Are beer drinkers patriotic?** Is the preferred beer of a continent typically a local brew? Is there a correlation, and if so, is it statistically significant or just a coincidence?

## **Additional datasets**

We don't need any additional datasets.

## **Methods**

**TASK 1: Data Pre-processing**

_Preprocessing Steps:_
* Duplicate removal, handling NaN values, and standardizing data formats.
* Location of beers, breweries and users corrections.
* Recomputation of beers ratings metrics to overcome high percentage of NaN values.

_Data charactersitics_: intial analysis on how data is distributed in the different datasets. 

_Outcome_: Clean, balanced datasets ready for temporal and demographic analysis. 

**TASK 2: Datasets distribution analysis**

Data from RateBeer and BeerAdvocate need to be processed to ensure correctness and organized flow of work. All work saved in `src/data`. The question now is if the two datasets can be merged or not.

_Statistical Approach_: Perform several test to compare distribution shapes and validate if the datasets can be merged.

- **Shapiro-Wilk Test**: Performed to determine whether the distribution of the variable *'avg'* follows a normal distribution.  
- **Q-Q Plots**: Used to visually assess the normality of the data.   
- **Kolmogorov-Smirnov (KS) Test**: Applied to compare the distributions of the two datasets and determine if they originate from the same distribution.  
- **Levene's Test**: Performed to check if the variances of the two datasets are equal before applying parametric tests like t-test or ANOVA.  

_Outcome_: A decision on merging data with normalized ratings and an understanding of dataset compatibility.

>If the tests and visual inspections reveal comparable distributions, we can confidently merge the datasets with normalized scores. If differences are notable, we could choose to keep the datasets separate and conduct analyses within each before combining higher-level results. 

**TASK 3: The Awards Begin - Identifying Annual and Continental Favorites**

_Approach_: Aggregate and rank beers by year, continent, and globally based on user ratings. Also merge ratings information with user and beer information, so as to analyze the demographic impact on the beer rating made by each user.

> US-specific Analysis: Separate analysis for the United States by state to explore intra-country variation (We are making this choice because the data from the US represents a clear majority. Our goal is to understand whether this majority is homogeneous or if there are further differences within it.).

_Subtasks_:
* Individual analysis on beers, breweries, users and ratings, taking into account parameters as location, time, and beer style, accordingly for each specific variable under analysis.

* Study on a merged dataframe, with information from beers, users, breweries and ratings. Analysis on location of beers, breweries and users. Time evolution of beer preferences from user and ratings information.

* Comparing result from the two datasets for deeper analysis.

_Outcome_: A structured dataset of top-rated beers annually and by continent.

**TASK 4: Aromatic Characteristics Extraction**

_NLP Techniques_: Implement NLP and topic modeling to identify common flavor descriptors from reviews in order to extract valuable characteristics for their aromatic profiling.

_Sentiment Analysis_: Measure sentiment for descriptive terms to capture preferences (e.g., “bitter” with positive sentiment indicating a preference).

_Longitudinal analysis_: We aim to understand how the characteristics of the beer perceived as the favorite evolve over time, in order to determine whether there is a common direction in taste preferences or if there are clear territorial distinctions.

_Prediction evaluation_: It will also be valuable to assess whether the extracted data leads to meaningful insights, and if we can develop a system capable of predicting the organoleptic traits that users will seek in the next year’s favorite beer.

_Outcome_: A database of characteristic descriptors for the preferred beer, grouped by year and region, with sentiment context.

**TASK 5: Make it visible!**

_Visualization Tools_: Consider tools like Tableau, Plotly, or D3.js for rich, interactive visualizations. Geographic visualizations can depict beer preferences by region and allow users to interact with data by year or flavor profile.

_Outcome_: Engaging, accessible visualizations that display the evolution of beer trends across the globe.

>Taking a small journey through the preferred tastes in different locations around the world!

## **Proposed timeline**

- **15.11.2024** – Task 1, Task 2 and starting Task 3
- **22.11.2024** – Task 3
- **29.11.2024** – Task 4 
- **15.12.2024** – Task 5


## **Organization within the team**

We strongly believe in teamwork and feel that ten eyes are always better than two. Although there were specific divisions of tasks (outlined below), we want to emphasize that we all contributed to every part of the project, advising and correcting each other along the way.

* **Ari** – Task 1, Task 2 and Task 3
* **Chiara** - ReadMe, Task 1 and Task 3
* **Jofre** – Task 3, Task 4 and Task 5
* **Jon** – Task 1 (mainly) and Task 2
* **Vittoria** – Task 2 and Task 3 (mainly)

## Requirements

The project requires the following Python libraries to be installed:

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computing.
- `matplotlib`: Data visualization.
- `seaborn`: Statistical data visualization.
- `scipy`: Scientific and statistical computing.
- `plotly`: Interactive data visualization.
- `geopandas`: Geospatial data analysis.

To install these dependencies, you can use the provided `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## **Repository structure**

The **current** directory structure of new project looks like this:

```
├── data                        <- Project data files (too big for GitHub)
│
├── src                         <- Source code
│   ├── data                            <- Data directory: processings notebooks
│        ├── pre_processing_.ipynb             <- Data pre-processing files for both datasets
│        └── merging_analysis.ipynb           <- Database beer merging analysis
│        └── utils.py           <- utils with functions for dataset analysis
│   ├── utils                            <- Utils directory: .py of results code
│        ├── utilities_beers_analysis.py        <- beers timeline analysis related functions
│        └── utilities_nlp_analysis.py          <- sentiment analysis related functions
│   ├── test                            <- Output of all tests done for graphics and plots
├── results.ipynb               <- initial analysis and results obtained
├── .gitignore                  <- List of files ignored by git
├── requirements.txt        <- File for installing python dependencies
└── README.md

```

## Questions for TAs

