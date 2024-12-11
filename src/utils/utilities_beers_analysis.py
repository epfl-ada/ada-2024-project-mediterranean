# utilities_beers_analysis.py
# ==============================
# Title: ADA Project Beer Analysis Utilities
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
import plotly.express as px
from iso3166 import countries_by_name




################################
###### INTERACTIVE PLOTS #######
################################

def plot_US_map_data(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_
    """

    beers_BA_US = dataset[dataset['location']=='United States']

    state_counts = beers_BA_US['location_region'].value_counts().reset_index()
    state_counts.columns = ['state', 'beer_count']
    

    # Map full state names to abbreviations
    us_state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    # Convert state names to abbreviations
    state_counts['state'] = state_counts['state'].map(us_state_abbrev)

    # Calculate log scale for better color differentiation
    state_counts['log_beer_count'] = np.log1p(state_counts['beer_count'])

    # Create an interactive map with Plotly for US states
    fig = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",  # Use 'USA-states' mode to restrict to US states
        color="log_beer_count",     # Log scale column for coloring
        hover_name="state",
        hover_data={
            "beer_count": True  # Display the original number of beers for clarity
        },
        title="Number of Beers per State in the US (Log Scale)"
    )

    # Update layout for better appearance and focus on the US
    fig.update_geos(
        scope="usa",  # Restrict to USA
        showcoastlines=True,
        coastlinecolor="Gray"
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar={
            'title': "Number of Beers (Log Scale)",
            'tickvals': [np.log1p(val) for val in [1, 10, 100, 1000, 10000, 100000]],  # Log scale ticks
            'ticktext': ['1', '10', '100', '1k', '10k', '100k']
        }
    )

    # Display the figure
    fig.show()
    #fig.write_html("us_beer_map.html")


def plot_world_map_data(dataset):
    """Creates a choropleth map for the whole world based on the dataset.

    Args:
        dataset (pd.DataFrame): Dataset containing country-level data with beer counts.
    """
    # Aggregate data by country
    country_counts = dataset['location'].value_counts().reset_index()
    country_counts.columns = ['country', 'beer_count']
    
    country_to_iso3 = {name.upper(): country.alpha3 for name, country in countries_by_name.items()}

    # Add a manual mapping for mismatched country names
    manual_country_mapping = {
        'United States': 'USA',
        'Russia': 'RUS',
        'United Kingdom': 'GBR',
        'South Korea': 'KOR',
        'Iran': 'IRN',
        'Czech Republic': 'CZE',
        'Georgia': 'GEO',
        # Add more mappings as necessary
    }

    # Normalize country names and apply manual mapping
    country_counts['iso_alpha3'] = country_counts['country'].map(manual_country_mapping).fillna(
        country_counts['country'].str.upper().map(country_to_iso3)
    )

    # Drop rows with missing ISO Alpha-3 codes
    country_counts = country_counts.dropna(subset=['iso_alpha3'])

    # Calculate log scale for better color differentiation
    country_counts['log_beer_count'] = np.log1p(country_counts['beer_count'])

    # Create an interactive map with Plotly for world data
    fig = px.choropleth(
        country_counts,
        locations="iso_alpha3",
        locationmode="ISO-3",  # Use ISO Alpha-3 codes for mapping
        color="log_beer_count",  # Log scale column for coloring
        hover_name="country",    # Display full country name on hover
        hover_data={
            "beer_count": True  # Display the original number of beers for clarity
        },
        title="Number of Beers per Country (Log Scale)"
    )

    # Update layout for better appearance
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="Gray"
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': "Number of Beers (Log Scale)",
            'tickvals': [np.log1p(val) for val in [1, 10, 100, 1000, 10000, 100000]],  # Log scale ticks
            'ticktext': ['1', '10', '100', '1k', '10k', '100k']
        }
    )

    # Display the figure
    fig.show()
    #fig.write_html("world_beer_map.html")


def plot_US_map_data_by_year(dataset):
    """Creates an interactive choropleth map for US states by year based on beer ratings.

    Args:
        dataset (pd.DataFrame): Dataset containing beer ratings per state and year.
    """

    # Aggregate data by state and year
    state_year_counts = dataset.reset_index()
    state_year_counts = state_year_counts.melt(
        id_vars=['year'], var_name='state', value_name='beer_count'
    )

    # Map full state names to abbreviations
    us_state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    # Add state abbreviations
    state_year_counts['state_abbrev'] = state_year_counts['state'].map(us_state_abbrev)

    # Drop rows with missing abbreviations
    state_year_counts = state_year_counts.dropna(subset=['state_abbrev'])

    # Calculate log scale for better color differentiation
    state_year_counts['log_beer_count'] = np.log1p(state_year_counts['beer_count'])

    # Create an interactive map with Plotly for US states
    fig = px.choropleth(
        state_year_counts,
        locations="state_abbrev",
        locationmode="USA-states",  # Use 'USA-states' mode to restrict to US states
        color="log_beer_count",     # Log scale column for coloring
        hover_name="state",
        hover_data={
            "beer_count": True,  # Display the original number of beers for clarity
            "log_beer_count": False
        },
        animation_frame="year",     # Animation by year
        title="Progression of Beer Ratings per State in the US (Log Scale)"
    )

    # Update layout for better appearance and focus on the US
    fig.update_geos(
        scope="usa",  # Restrict to USA
        showcoastlines=True,
        coastlinecolor="Gray"
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': "Number of Ratings (Log Scale)",
            'tickvals': [np.log1p(val) for val in [1, 10, 100, 1000, 10000, 100000]],  # Log scale ticks
            'ticktext': ['1', '10', '100', '1k', '10k', '100k']
        }
    )

    # Display the figure
    fig.show()


def plot_world_map_data_by_year(dataset):
    """Creates a choropleth map for the whole world based on the dataset with year-wise animation.

    Args:
        dataset (pd.DataFrame): Dataset containing country-level data with beer counts and years.
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from iso3166 import countries_by_name

    # Aggregate data by country and year
    country_year_counts = dataset.reset_index()
    country_year_counts = country_year_counts.melt(
        id_vars=['year'], var_name='country', value_name='beer_count'
    )

    # Mapping full country names to ISO Alpha-3 codes
    country_to_iso3 = {name.upper(): country.alpha3 for name, country in countries_by_name.items()}

    # Add a manual mapping for mismatched country names
    manual_country_mapping = {
        'United States': 'USA',
        'Russia': 'RUS',
        'United Kingdom': 'GBR',
        'South Korea': 'KOR',
        'Iran': 'IRN',
        'Czech Republic': 'CZE',
        'Georgia': 'GEO',
        # Add more mappings as necessary
    }

    # Normalize country names and apply manual mapping
    country_year_counts['iso_alpha3'] = country_year_counts['country'].map(manual_country_mapping).fillna(
        country_year_counts['country'].str.upper().map(country_to_iso3)
    )

    # Drop rows with missing ISO Alpha-3 codes
    country_year_counts = country_year_counts.dropna(subset=['iso_alpha3'])

    # Calculate log scale for better color differentiation
    country_year_counts['log_beer_count'] = np.log1p(country_year_counts['beer_count'])

    # Create an interactive map with Plotly for world data
    fig = px.choropleth(
        country_year_counts,
        locations="iso_alpha3",
        locationmode="ISO-3",  # Use ISO Alpha-3 codes for mapping
        color="log_beer_count",  # Log scale column for coloring
        hover_name="country",    # Display full country name on hover
        hover_data={
            "beer_count": True,  # Display the original number of beers for clarity
            "log_beer_count": False
        },
        animation_frame="year",  # Animation by year
        title="Progression of Beer Ratings per Country (Log Scale)"
    )

    # Update layout for better appearance
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="Gray"
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': "Number of Ratings (Log Scale)",
            'tickvals': [np.log1p(val) for val in [1, 10, 100, 1000, 10000, 100000]],  # Log scale ticks
            'ticktext': ['1', '10', '100', '1k', '10k', '100k']
        }
    )

    # Display the figure
    fig.show()
    fig.write_html("world_beer_map_by_year.html")