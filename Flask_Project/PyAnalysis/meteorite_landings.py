'''
Gavin Jampani
Section AF
meteorite_landings.py does various analytical operations on meteorite landings.
'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import io
from flask import send_file

def test_hello():
    return 'Hello World'

def timestamp_to_year(data):
    '''
    Takes in a pandas dataframe and returns an updated pandas dataframe with
    each value in the year column converted from a Floating TimeStamp object to
    a single integer value representing the year.
    '''
    data = data.dropna()

    try:
        data['year'] = data['year'].apply(lambda x: int(x[6:10]))
    except Exception as err:
        print("Error converting timestamp to YYYY: ", err)

    return data


def pd_to_gpd(data):
    '''
    Takes in a pandas dataframe and returns a new geopandas dataframe with a
    new geometry column called coordinates.
    '''
    data['coordinates'] = [Point(long, lat) for long, lat in
                           zip(data['reclong'], data['reclat'])]
    return gpd.GeoDataFrame(data, geometry='coordinates')


def merged_dataset(data, countries):
    '''
    Takes two geopandas dataframe, one for the meteorite landings dataset and
    one for the countries dataset, and returns the two merged into a single
    geopandas dataframe.
    '''
    data = data.dropna()
    merged = gpd.sjoin(countries, data, how='inner', op='intersects')
    return merged


def popular_class(data):
    '''
    Takes in a pandas dataframe and returns the name of the most popular class
    type. If there is a tie, returns the first class type that appears the
    most.
    '''
    series = data.groupby('recclass')['name'].count()
    return series.idxmax()


def popular_place_of_popular_class(merged, class_type):
    '''
    Takes a merged geodataframe and a name of a class type and returns the most
    common location of where the class type of meteorites passed land.
    '''
    merged = merged[merged['recclass'] == class_type]
    series = merged.groupby('CONTINENT')['recclass'].count()
    return series.idxmax()


def plot_popular_class(data, countries, class_type):
    '''
    Takes in two geopandas dataframe, one for the meteorite landings dataset
    and one for the countries dataset, and plots the popular class type on a
    world map.
    '''

    fig, ax1 = plt.subplots(1, figsize=(8, 8))
    countries.plot(ax=ax1)
    data = data[data['recclass'] == class_type]
    data.plot(ax=ax1, color='#78ffed')
    plt.title('Popular Meteorite Class')
    bytes_image = io.BytesIO()
    fig.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_popular_class.png',
                     mimetype='image/png')


def rarest_class(data):
    '''
    Takes in a pandas dataframe and returns the name of the rarest class type.
    If there is a tie, returns the first class type that appears the most.
    '''
    series = data.groupby('recclass')['name'].count()
    return series.idxmin()


def plot_rarest_class(data, countries, class_type):
    '''
    Takes in two geopandas dataframe, one for the meteorite landings dataset
    and one for the countries dataset, and plots the rarest class type on a
    world map.
    '''

    fig, ax1 = plt.subplots(1, figsize=(8, 8))
    countries.plot(ax=ax1)
    data = data[data['recclass'] == class_type]
    data.plot(ax=ax1, color='#78ffed')
    plt.title('Rarest Meteorite Class')
    bytes_image = io.BytesIO()
    fig.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_rarest_class.png',
                     mimetype='image/png')


def avg_mass(data):
    '''
    Takes a pandas dataframe and returns a float representing the average mass
    of the meteorites.
    '''
    return data['mass (g)'].mean()


def range_mass(data):
    '''
    Takes a pandas dataframe and returns a float representing the range of
    mass of the meteorites.
    '''
    data = data.dropna()
    return (data['mass (g)'].max()) - (data['mass (g)'].min())


def biggest_mass(data):
    '''
    Takes a pandas dataframe and returns a float representing the biggest mass
    of meteorite.
    '''
    data = data.dropna()
    return data['mass (g)'].max()


def smallest_mass(data):
    '''
    Takes a pandas dataframe and returns a float representing the biggest mass
    of meteorite.
    '''
    data = data.dropna()
    return data['mass (g)'].min()


def plot_mass_overtime(data):
    '''
    Takes in a pandas dataframe and plots a line plot displaying the mass of
    the meteorites overtime.
    '''
    data = data.dropna()
    data = data[(data['year'] >= 1400) & (data['year'] <= 2000)]
    sns.relplot(x='year', y='mass (g)', kind='line', ci=None, data=data)
    plt.title('Mass of Meteorites Overtime')
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png', bbox_inches='tight')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_mass_overtime.png',
                     mimetype='image/png')


def plot_all_mass_map(data, countries):
    '''
    Takes two geopandas, one of the dataset and one of the countries dataset,
    and plots all the meteorites on a world map based on their mass.
    '''

    fig, ax1 = plt.subplots(1, figsize=(8, 8))
    countries.plot(ax=ax1)
    data = data.dropna()
    data.plot(ax=ax1, column='mass (g)', legend=True)
    plt.title('All Meteorites Based on Mass')
    bytes_image = io.BytesIO()
    fig.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_all_masses.png',
                     mimetype='image/png')


def plot_bigger_mass_map(data, countries):
    '''
    Takes two geopandas, one of the dataset and one of the countries dataset,
    and plots the biggest meteorites on a world map based on their mass.
    '''
    fig, ax1 = plt.subplots(1, figsize=(8, 8))
    countries.plot(ax=ax1)
    data = data.dropna()
    data = data[data['mass (g)'] >= 1000000]
    data.plot(ax=ax1, column='mass (g)', legend=True)
    plt.title('Biggest Meteorites Based on Mass')
    bytes_image = io.BytesIO()
    fig.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_bigger_masses.png',
                     mimetype='image/png')

def predict_place_of_meteorite_impact(merged):
    '''
    Takes in a merged geopandas dataframe and trains and tests a model to
    predict where a meteorite is most likely to land.
    '''
    # merged = merged.loc[:, ['CONTINENT', 'recclass', 'mass (g)', 'year']]
    merged = merged.loc[:, ['CONTINENT', 'mass (g)', 'year']]
    merged = merged.dropna()
    X = merged.loc[:, merged.columns != 'CONTINENT']
    y = merged.loc[:, 'CONTINENT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # model = DecisionTreeClassifier()
    model = rf()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    return accuracy_score(y_test, model.predict(X_test))

def predict(mass, year):
    model = joblib.load('model.pkl')
    
    query_df = pd.DataFrame({'mass (g)': mass, 'year': year}, index=[1])

    prediction = model.predict(query_df)
    prediction = str(prediction)
    return prediction[2:len(prediction)-2]

def main():
    sns.set()
    pdata = pd.read_csv('Meteorite_Landings.csv', na_values=[])
    pdata = timestamp_to_year(pdata)
    gdata = pd_to_gpd(pdata)
    countries = gpd.read_file('ne_110m_admin_0_countries.shp')
    merged = merged_dataset(gdata, countries)

    '''
    print('Popular Class Type: ')
    print(popular_class(pdata))
    print('Popular Class Type Location: ')
    print(popular_place_of_popular_class(merged, 'L6'))
    plot_popular_class(gdata, countries, 'L6')

    print('Rarest Class Type: ')
    print(rarest_class(pdata))
    plot_rarest_class(gdata, countries, 'Acapulcoite/lodranite')

    print('Average Mass: ')
    print(avg_mass(pdata))
    print('Range of Mass: ')
    print(range_mass(pdata))
    print('Biggest Mass: ')
    print(biggest_mass(pdata))
    print('Smallest Mass: ')
    print(smallest_mass(pdata))
    plot_mass_overtime(pdata)

    plot_all_mass_map(gdata, countries)
    plot_bigger_mass_map(gdata, countries)
    '''


if __name__ == '__main__':
    main()
