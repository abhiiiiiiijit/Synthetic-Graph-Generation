import pandas as pd

def main():
    #get the city population data
    df_city_p = pd.read_csv('../data/city_population',delimiter=';')
    print(df_city_p.head())


if __name__ == 'main':
    main()