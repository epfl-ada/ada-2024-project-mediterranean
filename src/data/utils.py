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
from collections import Counter
import math
import numpy as np
from datetime import datetime


########################## PREPROCESSING BLOCK #############################

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

def calculate_style_score(row, style_mean, m, c):
    N = row['nbr_ratings']
    avg = row['avg_scaled']
    C = style_mean[row['style']] if row['style'] in style_mean else c
    return np.round((N / (N + m)) * avg + (m / (N + m)) * C,2)

def calculate_overall_score(row, c, m):
    N = row['nbr_ratings']
    avg = row['avg_scaled']
    C = c
    return np.round((N / (N + m)) * avg + (m / (N + m)) * C,2)


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

def plotting_dist(dist, dataset, n=15, location='location'):   
    # Select only the first n (to make the plot readable)
    dist_bar = dist.sort_values(by='brewery_count', ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(dist_bar[location], dist_bar['brewery_count'], color='skyblue',label='Number of breweries')
    ax.set_title(f'Number of breweries for each country {dataset} dataset - top {n}', fontsize=15)
    ax.set_xlabel('Number of Breweries', fontsize=15)
    ax.set_ylabel('Country', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)  # Tick dimension
    ax.tick_params(axis='y', labelsize=12)  
    plt.errorbar(dist_bar['brewery_count'], dist_bar[location], 
                 xerr=dist_bar['std_beers'],  # std
                 fmt='o', 
                 color='orange',  
                 label='Standard Deviation',  
                 capsize=5)  
    # Make the table
    table_data = dist_bar[[location, 'mean_beers']]
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
def comparing_plot(dist_BA, dist_RB, dist_us_BA, dist_us_RB,n=15, location='location', location_region='location_region'):
    dist_BA_bar = dist_BA.sort_values(by='brewery_count', ascending=False).head(n)
    dist_RB_bar = dist_RB.sort_values(by='brewery_count', ascending=False).head(n)
    dist_us_BA_bar = dist_us_BA.sort_values(by='brewery_count', ascending=False).head(n)
    dist_us_RB_bar = dist_us_RB.sort_values(by='brewery_count', ascending=False).head(n)
    dataset_bar = [dist_BA_bar, dist_RB_bar, dist_us_BA_bar, dist_us_RB_bar]
    actual_loc = [location, location, location_region, location_region]
    title = ['BA COMPLETE DATASET', 'RB COMPLETE DATASET','BA US DATASET', 'RB US DATASET']
    
    i = 0
    fig, ax = plt.subplots(2,2, figsize=(20,17)) 
    for i in range(4):
        data = dataset_bar[i]
        loc = actual_loc[i]
        sbplt = ax[i // 2, i % 2]
        sbplt.barh(data[loc], data['brewery_count'], color='skyblue',label='Number of breweries')
        sbplt.legend()
        sbplt.set_title(title[i], fontsize=15)
        sbplt.errorbar(data['brewery_count'], data[loc], 
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

def check_differences(dataset, processed_dataset,name, name_processed):
    print('The starting dimensions of dataset', name,'was:', dataset.shape)
    print('Columns:', dataset.columns)
    print('The processed dataset', name_processed,' has dimensions:', processed_dataset.shape)
    print('Columns:', processed_dataset.columns)
    return None


def nbr_user_per_location_per_time(users, dataset_name):
    """
        users  = dataframe of users (processed one)
        name   = the name of dataset
        return = the function returns a table with the years as indexes
                 and the various locations we have as columns. For each row column pair we have the number of platform users per location 
    """
    users['joined'] = pd.to_datetime(users['joined'])
    users['year_joined'] = users['joined'].dt.year
    # Group by year_joined and location and then count only the uniqur users
    user_counts = users.groupby(['year_joined', 'location'])['user_id'].nunique().reset_index()
    # Create a pivot table where the index is "year_joined", the columns are “location”, and the values are user counts
    user_distribution_by_year_location = user_counts.pivot_table(index='year_joined', columns='location', values='user_id', fill_value=0)
    # Apply the cumulative user count -> This is important because I'm intrested in understanding the distribution of user per year,
    # not how many new user join the platform for every year
    user_distribution_by_year_location = user_distribution_by_year_location.cumsum()
    # Total users by year (sum of all locations)
    user_distribution_by_year_location['Total'] = user_distribution_by_year_location.sum(axis=1)
    return user_distribution_by_year_location


def time_machine(table_users, dataset_name, location, color='b', scale='linear'):
    """
        table_users  =  is the table that has yeas as index and location as columns. 
                        In every cell we have the number of user for each location per year
                        (output of nbr_user_per_location_per_time)
        dataset_name =  name of the dataset
        location     =  the location we want to visualize (the column of the table)
        color        =  the color that I prefere for the plot
        scale        =  the scale you prefer for the plotting
        output = plot that show how the subscription change over time for that site and for that country
    """
    plt.plot(table_users.index, table_users[location],marker='o', color=color, linestyle='-', label=f'Cumulative annual memberships of {location}')
    plt.yscale(scale)
    # Add title and labels
    plt.title(f'Trend of annual site registrations in dataset {dataset_name}')
    plt.xlabel('Years')
    plt.legend()
    plt.ylabel('Subscription')

# This function create a dinamic and interactive plot in which you can see how the users' subscription change in number year by year. You can move the slider and appreciate the variation for the selected n locations
def plot_user_per_location_per_time_interactive(table_users, dataset_name, n=15):
    """
    table_users  = is the table that has yeas as index and location as columns. 
                   In every cell we have the number of user for each location per year
                   (output of nbr_user_per_location_per_time)
    dataset_name = name of the current investigate  dataset
    n            = number of top-n location per number of user to show (this is just to make the visualization more clear)
    
    """
    # Set render
    pio.renderers.default = "browser"
    # List with years in my dataset (only the one that are unique) [is just a double check, we have already set unique before]
    years = table_users.index.tolist()

    # Create general figure
    fig = go.Figure()

    # Add a trace for each year (using the slider to show only the selected graph)
    for year in years:
        # Extract data for current year and drop column of total
        data = table_users.loc[year].drop('Total')
    
        # Select just the first 15 location for number of users (to make visualization more clear)
        data_top_n = data.nlargest(n)
    
        # Add the bar graph for this year, showing only the first n locations
        fig.add_trace(go.Bar(
            x=data_top_n.index,
            y=data_top_n.values,
            name=f"Year {year}",
            visible=(year == years[0])  # Only the first year is initially visible
        ))

    # Configure the slider steps for each year
        # Each step configures which chart (year) to show when the user interacts with the slider
        # When a user changes the year on the slider, the step associated with that year indicates which bar graph
        # should become visible.
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"User Distribution by Location for Year {year}"}],
            label=str(year)  # Set the step label as the current year
        )
        # Make visible only the graph for the current year
        step["args"][0]["visible"][i] = True
        steps.append(step)

    # Configure the slider and add it to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    # Final Layout Configuration
    fig.update_layout(
        sliders=sliders,
        title="User Distribution by Location for Each Year (Top n Locations)",
        xaxis_title="Location",
        yaxis_title="Number of Users"
    )

    fig.show()


# Static and cumulative plot version of plot_user_per_location_per_time_interactive. (Side Note: In the for loop, I go from the last to the first year so that the graphs are layered from highest to lowest. In a forward order, the last year would cover the previous ones, making it harder to appreciate changes over time. This approach works because, over the years, the number of subscribers can only increase or remain the same.)
def plot_user_per_location_per_time_cumulative(table_users, dataset_name, n=10):
    """
    table_users  = is the table that has years as index and location as columns. 
                   In every cell we have the number of user for each location per year
    dataset_name = name of the current investigate dataset
    n            = number of top-n locations per number of users to show (this is just to make the visualization clearer)
    """
    years = table_users.index.tolist()

    # Step 1: Count how many times each location appears in the top-N of each year
    top_n_locations_count = Counter()

    # Iterate over the years and count top-N locations
    for year in years:
        # Get the top-n locations for this year (exclude 'Total' column)
        data = table_users.loc[year].drop('Total')
        top_n = data.nlargest(n).index  # Get the top-N locations for the current year
        top_n_locations_count.update(top_n)  # Update the count for these locations
        
    # Step 2: Sort locations based on their frequency (most frequent locations first)
    location_order = [location for location, count in top_n_locations_count.most_common()]
    location_order = location_order[:n]
    # Print the top-N locations sorted
    print(f"Top {n} locations: {list(location_order[:n])}")
    # Step 3: Create the figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Create a color map with as many colors as years
    colors = plt.cm.get_cmap('tab10', len(years))  # Using 'tab10' colormap 
    
    # Step 4: Iterate over each year and plot cumulative data
    handles = []  # To store legend handles
    for i, year in enumerate(reversed(years)):
        # Extract data for the current year and drop the 'Total' column
        data = table_users.loc[year].drop('Total')
        data_top_n = data.loc[location_order].nlargest(n)  # Get top-N based on pre-defined location order
        x = data_top_n.index
        y = data_top_n.values
        bars = ax.bar(x, y, color=colors(i), width=0.4, align='center')

        # Create a legend handle for this year
        handles.append(bars[0])

    # Step 5: Add a legend with the colors corresponding to each year
    ax.legend(handles=handles, labels=years[::-1], title="Years")
    
    # Set the title and labels
    ax.set_title(f"Cumulative User Distribution by Location for Each Year ({dataset_name})")
    ax.set_xlabel("Location")
    ax.set_ylabel("Cumulative Number of Users")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Step 6: Show the plot
    plt.show()
    return location_order[:n]


def find_top_n(table_users, n=7):
    """
    This function returns the locations that have been in the top-n for the number of users across all the years stored in the database.
    """
    years = table_users.index.tolist()
    # Step 1: Count how many times each location appears in the top-N of each year
    top_n_locations_count = Counter()
    # Iterate over the years and count top-N locations
    for year in years:
        # Get the top-n locations for this year (exclude 'Total' column)
        data = table_users.loc[year].drop('Total')
        top_n = data.nlargest(n).index  # Get the top-N locations for the current year
        top_n_locations_count.update(top_n)  # Update the count for these locations
        
    # Step 2: Sort locations based on their frequency (most frequent locations first)
    location_order = [location for location, count in top_n_locations_count.most_common()]
    location_order = location_order[:n]
    # Print the top-N locations sorted
    print(f"Top {n} locations: {list(location_order[:n])}")
    return location_order



################################################
################# BREWER BLOCK #################
################################################


########################## FILTERING BLOCK #############################
# This function remove all the line where the brewer has 0 nbr of beer
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
##########################################################################


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


def loc_distribution(breweries, dataset, groupping_column):
    print('Dataset:', dataset)
    print('- Number of unique "location" value in the dataset:', breweries['location'].nunique())
    distribution = breweries.groupby(groupping_column).agg(
        brewery_count=('location', 'size'),
        total_beers=('nbr_beers', 'sum'),
        mean_beers=('nbr_beers', 'mean'),
        std_beers=('nbr_beers', 'std'),
        median_beers=('nbr_beers', 'median')).reset_index()
    #distribution['mean_beers'] = distribution['mean_beers'].round(2)
    #distribution['median_beers'] = distribution['median_beers'].round(2)
    return distribution



################################################
################## BEER BLOCK ##################
################################################

def calculate_weighted_avg(row, global_mean,m):
    n = row['nbr_ratings']
    avg = row['avg']
    return np.round(((avg*n)+ (global_mean*m))/(n+m),3)

def tripe_hist_plot_beers(top_avg_beers, top_overall_beers, top_style_beers, title):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.barplot(data=top_avg_beers, x='avg', y='beer_name', ax=axes[0],  legend=False)
    axes[0].set_xlim(0.5, 1)
    axes[0].set_title("Top 20 Beers by Average Rating")
    axes[0].set_xlabel("Average Rating")
    axes[0].set_ylabel("Beer Name")

    sns.barplot(data=top_overall_beers, x='Overall_score', y='beer_name', ax=axes[1],  legend=False)
    axes[1].set_xlim(50, 100)
    axes[1].set_title("Top 20 Beers by Overall Score")
    axes[1].set_xlabel("Overall Score")
    axes[1].set_ylabel("")

    sns.barplot(data=top_style_beers, x='Style_score', y='beer_name', ax=axes[2],  legend=False)
    axes[2].set_xlim(50, 100)
    axes[2].set_title("Top 20 Beers by Style Score")
    axes[2].set_xlabel("Style Score")
    axes[2].set_ylabel("")

    plt.tight_layout()
    plt.show()

def tripe_hist_plot_beers_weighted(top_avg_beers, top_overall_beers, top_style_beers, title):
        # Set the style of the plots
    sns.set_theme(style="whitegrid")

    # Create a figure for 3 separate plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot top beers by avg
    sns.barplot(data=top_avg_beers, x='wavg', y='beer_name', ax=axes[0], legend  = False)
    axes[0].set_xlim(0.5,1)  # Adjust x-axis limits for avg
    axes[0].set_title("Top 20 Beers by Weighted Average Rating")
    axes[0].set_xlabel("Weighted Average Rating")
    axes[0].set_ylabel("Beer Name")

    # Plot top beers by overall score
    sns.barplot(data=top_overall_beers, x='Overall_score', y='beer_name', ax=axes[1], legend  = False)
    axes[1].set_xlim(50,100)
    axes[1].set_title("Top 20 Beers by Overall Score")
    axes[1].set_xlabel("Overall Score")
    axes[1].set_ylabel("")

    # Plot top beers by style score
    sns.barplot(data=top_style_beers, x='Style_score', y='beer_name', ax=axes[2], legend  = False)
    axes[2].set_xlim(50,100)
    axes[2].set_title("Top 20 Beers by Style Score")
    axes[2].set_xlabel("Style Score")
    axes[2].set_ylabel("")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def tripe_hist_plot_beers_weighted2(top_avg_beers, top_overall_beers, top_style_beers, title):
    # Set the style of the plots
    sns.set_theme(style="whitegrid")

    # Create a figure for 3 separate plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot top beers by avg
    sns.barplot(data=top_avg_beers, x='wavg', y='beer_label', ax=axes[0], legend = False)
    axes[0].set_xlim(0.5, 1)
    axes[0].set_title("Top 20 Beers by Weighted Average Rating")
    axes[0].set_xlabel("Average Rating")
    axes[0].set_ylabel("Beer Name (Location)")

    # Plot top beers by overall score
    sns.barplot(data=top_overall_beers, x='Overall_score', y='beer_label',ax=axes[1], legend=False)
    axes[1].set_xlim(50, 100)
    axes[1].set_title("Top 20 Beers by Overall Score")
    axes[1].set_xlabel("Overall Score")
    axes[1].set_ylabel("")  # Remove duplicate ylabel for cleaner layout

    # Plot top beers by style score
    sns.barplot(data=top_style_beers, x='Style_score', y='beer_label', ax=axes[2], legend=False)
    axes[2].set_xlim(50, 100)
    axes[2].set_title("Top 20 Beers by Style Score")
    axes[2].set_xlabel("Style Score")
    axes[2].set_ylabel("")  # Remove duplicate ylabel for cleaner layout

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()