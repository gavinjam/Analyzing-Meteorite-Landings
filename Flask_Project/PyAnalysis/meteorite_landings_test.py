'''
Gavin Jampani
Section AF
meteorite_landings_test.py tests the calculation functions in
meteorite_landings.py.
'''
import meteorite_landings as ml
import pandas as pd
import geopandas as gpd
from cse163_utils import assert_equals


def test_popular_class(data):
    '''
    Tests the popular_class function.
    '''
    print('====Testing the popular_class function====')
    assert_equals('EH4', ml.popular_class(data))


def test_popular_place_of_popular_class(merged, class_type):
    '''
    Tests the popular_place_of_popular_class function.
    '''
    print('====Testing the popular_place_of_popular_class function====')
    assert_equals('Asia',
                  ml.popular_place_of_popular_class(merged,
                                                    class_type))


def test_rarest_class(data):
    '''
    Tests the rarest_class function.
    '''
    print('====Testing the rarest_class function====')
    assert_equals('Acapulcoite', ml.rarest_class(data))


def test_avg_mass(data):
    '''
    Tests the avg_mass function.
    '''
    print('====Testing the avg_mass function====')
    assert_equals(13604.0, ml.avg_mass(data))


def test_range_mass(data):
    '''
    Tests the range_mass function.
    '''
    print('====Testing the range_mass function====')
    assert_equals(106979.0, ml.range_mass(data))


def test_biggest_mass(data):
    '''
    Test the biggest_mass function.
    '''
    print('====Testing the biggest_mass function====')
    assert_equals(107000.0, ml.biggest_mass(data))


def test_smallest_mass(data):
    '''
    Test the smallest_mass function.
    '''
    print('====Testing the smallest_mass function====')
    assert_equals(21, ml.smallest_mass(data))


def main():
    pdata = pd.read_csv('Meteorite_Landings.csv', na_values=[])
    pdata = pdata.loc[:10, :]
    pdata = ml.timestamp_to_year(pdata)
    gdata = ml.pd_to_gpd(pdata)
    countries = gpd.read_file('ne_110m_admin_0_countries.shp')
    merged = ml.merged_dataset(gdata, countries)

    test_popular_class(pdata)
    test_popular_place_of_popular_class(merged, 'EH4')
    test_rarest_class(pdata)
    test_avg_mass(pdata)
    test_range_mass(pdata)
    test_biggest_mass(pdata)
    test_smallest_mass(pdata)


if __name__ == '__main__':
    main()
