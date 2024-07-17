import pandas as pd
from geopy.geocoders import Nominatim
import time

def main():
    #get the city population data

    df_city_p = pd.read_csv('./data/city_population.csv',delimiter=';'
                            ,encoding='utf-8',header=None)
    
    #drop other columns, only take the 2023 data
    df_city_p = df_city_p.drop(columns=[0,2,3,4,5])

    #change column name
    col_name = {1:'city', 6:'population'}

    df_city_p.rename(columns=col_name,inplace=True)





    # change data type of population to int
    df_city_p['population'] = df_city_p['population'].astype('Int64')

    #Consider only the cities with population more than 100000
    df_city_p = df_city_p[df_city_p['population']>=100000]

    # extract only the city name
    # df_city_p['city'] = df_city_p['city'].apply(lambda x: get_city_from_df(x, ',', 0))
    df_city_p['city'] = df_city_p['city'].str.split(',',expand=True)[0]
    # df_city_p['city'] = df_city_p['city'].apply(decode_value)
    # df_city_p[1] = df_city_p[1].apply(decode_value)
    # df_city_p['city'] = df_city_p['city'].encode('latin1')

    # print(df_city_p.head(10))
    # print("Hello")
    ########################


    # Initialize the geolocator
    geolocator = Nominatim(user_agent="city_locator")

    # List of cities
    cities = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt"]

    # Function to get latitude and longitude
    def get_lat_lon(city):
        try:
            location = geolocator.geocode(city + ", Germany")
            if location:
                return (location.latitude, location.longitude)
            else:
                return None
        except Exception as e:
            print(f"Error getting coordinates for {city}: {e}")
            return None

    # Iterate through the cities and get their coordinates
    city_coords = {}
    for city in cities:
        coords = get_lat_lon(city)
        if coords:
            city_coords[city] = coords
        time.sleep(1)  # To avoid hitting the API rate limit

    # Print the results
    for city, coords in city_coords.items():
        print(f"{city}: Latitude = {coords[0]}, Longitude = {coords[1]}")


def decode_value(value):
    try:
        return value.encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value

def get_city_from_df(value, delimiter, part_index):
    try:
        parts = value.split(delimiter)
        return parts[part_index].strip() if len(parts) > part_index else None
    except Exception as e:
        print(f"Error processing value '{value}': {e}")
        return None

if __name__ == '__main__':
    main()