# utils.py
# ==============================
# Title: ADA Project Data Processing Utilities
# ==============================

import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import seaborn as sns
import scipy.stats as stats


def dataset_information(dataset, dataset_name):
    """Automatic dataset information retrieval

    Args:
        dataset (pandas.core.frame.DataFrame): pandas dataset
        dataset_name (str): name for output information

    Returns:
        None
    """

    print('\n')
    print('########################################################')
    print('We are starting analysing dataset', dataset_name)
    print('- Dimension of starting dataset:', dataset.shape)
    print('- Columns of dataset: ', dataset.columns)
    print('- Are all the id unique? Answer:', dataset[dataset.columns[0]].is_unique)
    print('- Are there some values that are NaN inside the dataset? Answer:',dataset.isna().any().any())
    print('Head: \n', dataset.head())
    return None

def ensure_col_types(dataset, cols_int, cols_float, cols_string):
    """_summary_

    Args:
        dataset (pandas.core.frame.DataFrame): _description_
        cols_num (str): _description_
        cols_string (str): _description_

    Returns:
        None
    """
    dataset[cols_int]  = dataset[cols_int].astype(int)
    dataset[cols_float]  = dataset[cols_float].astype(float)
    dataset[cols_string]  = dataset[cols_string].astype(str)
    return None

def ensure_col_types2(dataset, cols_int, cols_float, cols_string):
    """
    Ensures that specified columns in the dataset have the correct data types.
    
    Args:
        dataset (pandas.DataFrame): The DataFrame to process.
        cols_int (list): List of columns to convert to integers.
        cols_float (list): List of columns to convert to floats.
        cols_string (list): List of columns to convert to strings.
    
    Returns:
        pandas.DataFrame: The updated DataFrame with corrected column types.
    """
    # Convert columns to integer, handling errors
    for col in cols_int:
        if col in dataset.columns:
            try:
                dataset[col] = pd.to_numeric(dataset[col], errors='coerce').astype('Int64')
            except Exception as e:
                print(f"Error converting {col} to int: {e}")

    # Convert columns to float, handling errors
    for col in cols_float:
        if col in dataset.columns:
            try:
                dataset[col] = pd.to_numeric(dataset[col], errors='coerce').astype(float)
            except Exception as e:
                print(f"Error converting {col} to float: {e}")

    # Convert columns to string, handling errors
    for col in cols_string:
        if col in dataset.columns:
            try:
                dataset[col] = dataset[col].astype(str)
            except Exception as e:
                print(f"Error converting {col} to string: {e}")

    return dataset


def write_csv_into_directory(output_dir, output_file, dataset):
    """_summary_

    Args:
        output_dir (_type_): _description_
        output_file (_type_): _description_
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    dataset.to_csv(file_path, index=False)
    print(f'Dataset successfully saved to {file_path}')
    return None

def remove_zero_beer(breweries, dataset):
    print('FILTER BLOCK')
    print('-- The dimension of starting dataset',dataset,':', breweries.shape[0])
    breweries = breweries.loc[~(breweries['nbr_beers']==0)]
    print('-- The dimension of filtered dataset',dataset,':', breweries.shape[0])
    return breweries
    
# Cleaning 'location' column from HTML entries
def clean_location(location):
    # Remove HTML 
    return re.sub(r'<[^>]+>', '', location).strip()

# We clean the data dropping the entities that are not in the desided format
def clean_unfitt_data(breweries):
    # We create a mask that select all rows where location column starts with 'United States'
    mask = breweries['location'].str.startswith('United States')
    # We create a mask that consider only the items has 'United States' followed by ', letters' (valid entries) 
    valid_us_mask = breweries['location'].str.match(r'^United States, [A-Za-z\s]+$')
    # The filtered dataset has all the rows that haven't in location 'United States,..' + the ones about US in valid format
    # Mantieni le righe che non sono statunitensi o che sono nel formato corretto
    filtered_breweries = breweries[~mask | valid_us_mask]
    return filtered_breweries

def preprocess_location(df, location_column='location'):
    """
    Preprocess the location column in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the location column.
    location_column (str): The name of the location column to preprocess.
    
    Returns:
    pd.DataFrame: The DataFrame with the preprocessed location column.
    """
    # Convert to lowercase and remove leading/trailing whitespace
    df[location_column] = df[location_column].str.lower().str.strip()
    
    # Replace state-specific locations with 'United States'
    df[location_column] = df[location_column].apply(lambda x: 'united states' if isinstance(x, str) and 'united states' in x else x)
    
    # Correct common misspellings and variations
    # These corrections come from manual inspection of the data and checking that same country is instatiated differently in each dataset
    location_corrections = {
        'england': 'united kingdom',
        'northern ireland': 'united kingdom',
        'scotland': 'united kingdom',
        'wales': 'united kingdom',
        'czech republic': 'czechia',
        'slovak republic': 'slovakia',
    }
    
    df[location_column] = df[location_column].replace(location_corrections)
    
    # Handle missing values by removing rows with missing locations
    df = df.dropna(subset=[location_column])
    
    return df
##########################################################################

# Extracting data of US from dataset
def us_extraction(breweries, dataset):
    # We filter the rows when the 'location' column starts with 'United Stater,' 
    us_breweries = breweries[breweries['location'].str.startswith('United States,')].copy()
    # Create a new column 'state' with what is after the ,
    us_breweries['state'] = us_breweries['location'].str.split(',').str[1].str.strip()
    # Now column 'location' is  useless, so we drop it
    us_breweries = us_breweries.drop(columns='location')
    us_breweries = us_breweries.rename(columns={'state': 'location'}) #useful for reusing other function
    # Give a name to this new dataset (useful for other function)
    dataset_name = 'US_'+dataset
    return us_breweries, dataset_name

# Location distribution analysis: this function group data by location and show:
# - how many breweries we have for each country 
# - how many beers we have per country in total
# - mean of beers we have for each contry in every breweries
# - std of beers we have for each contry median of beers we have for each contry in every breweries ###
# - median of beers we have for each contry in every breweries
def loc_distribution(breweries, dataset):
    # we uniform the values in column location
    breweries['location'] = breweries['location'].apply(clean_location) #filter function
    breweries = clean_unfitt_data(breweries) #filter function
    print('Dataset:', dataset)
    print('- Number of unique "location" value in the dataset:', breweries['location'].nunique())
    distribution = breweries.groupby('location').agg(
        brewery_count=('location', 'size'),
        total_beers=('nbr_beers', 'sum'),
        mean_beers=('nbr_beers', 'mean'),
        std_beers=('nbr_beers', 'std'),
        median_beers=('nbr_beers', 'median')).reset_index()
    #distribution['mean_beers'] = distribution['mean_beers'].round(2)
    #distribution['median_beers'] = distribution['median_beers'].round(2)
    return distribution

########################## PLOTTING BLOCK #############################
# Plotting_dist funtion plot the number of brewer for the best n (=15) 
# countries. We can visualize the standard deviation and also the mean
# number of beer in brewer for each country

def plotting_dist(dist, dataset,title, n=15):   
    # Select only the first n (to make the plot readable)
    dist_bar = dist.sort_values(by='brewery_count', ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(dist_bar['location'], dist_bar['brewery_count'], color='skyblue',label='Number of breweries')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Number of Breweries', fontsize=15)
    ax.set_ylabel('Country', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)  # Tick dimension
    ax.tick_params(axis='y', labelsize=12)  
    plt.errorbar(dist_bar['brewery_count'], dist_bar['location'], 
                 xerr=dist_bar['std_beers'],  # std
                 fmt='o', 
                 color='orange',  
                 label='Standard Deviation',  
                 capsize=5)  
    # Make the table
    table_data = dist_bar[['location', 'mean_beers']]
    table_data['mean_beers']=round(table_data['mean_beers'],2) # We round the data for a better readability
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='right',
                     bbox=[1.05, 0, 0.5, 1])  # [left, bottom, width, height]
    # Table setting
    table_data = table_data.rename(columns={'mean_beers': 'Mean Beers for Brewer','location': 'Location'})
    table.auto_set_font_size(False) # Disable automatic font resizing
    table.set_fontsize(10)  
    table.scale(1, 1.5) 
    plt.show()

def comparing_plot(dist_BA, dist_RB, dist_us_BA, dist_us_RB,n=15):
    dist_BA_bar = dist_BA.sort_values(by='brewery_count', ascending=False).head(n)
    dist_RB_bar = dist_RB.sort_values(by='brewery_count', ascending=False).head(n)
    dist_us_BA_bar = dist_us_BA.sort_values(by='brewery_count', ascending=False).head(n)
    dist_us_RB_bar = dist_us_RB.sort_values(by='brewery_count', ascending=False).head(n)
    dataset_bar = [dist_BA_bar, dist_RB_bar, dist_us_BA_bar, dist_us_RB_bar]
    title = ['BA COMPLETE DATASET', 'RB COMPLETE DATASET','BA US DATASET', 'RB US DATASET']
    
    i = 0
    fig, ax = plt.subplots(2,2, figsize=(20,11),sharex=True)
    for i in range(4):
        data = dataset_bar[i]
        sbplt = ax[i // 2, i % 2]
        sbplt.barh(data['location'], data['brewery_count'], color='skyblue',label='Number of breweries')
        sbplt.legend()
        sbplt.set_title(title[i], fontsize=15)
        sbplt.errorbar(data['brewery_count'], data['location'], 
                     xerr=data['std_beers'],  # std
                     fmt='o', 
                     color='orange',  
                     label='Standard Deviation',  
                     capsize=5)
        sbplt.set_xlabel('Number of Breweries', fontsize=15)
        sbplt.set_ylabel('Country', fontsize=15)
        sbplt.tick_params(axis='x', labelsize=12)  # Tick dimension
        sbplt.tick_params(axis='y', labelsize=12) 

    return None



def location_into_loc_region(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataset[['location', 'location_region']] = dataset['location'].str.split(',', n=1, expand=True)

    return dataset

def update_location_and_region(dataset):
    uk_regions = ['England', 'Scotland', 'Wales', 'Northern Ireland']
    
    # Check if the location is part of the UK regions
    for i in range(len(dataset)):
        region=dataset.loc[i,'location']
        if region in uk_regions:
            dataset.loc[i,'location'] = 'United Kingdom'  # Set the location to 'UK'
            dataset.loc[i,'location_region'] = region  # Set the region to the specific UK region

    return dataset


def plot_normality(data, title):
    # Set up the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Histogram
    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {title}")
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q plot of {title}")
    
    # Box plot
    sns.boxplot(data, ax=axes[2])
    axes[2].set_title(f"Box plot of {title}")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

    stat1, p1 = stats.shapiro(data)
    print(f"Shapiro Test for Dataset 1: Statistic = {stat1}, p-value = {p1}")