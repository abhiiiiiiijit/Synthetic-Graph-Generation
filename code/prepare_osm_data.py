import pandas as pd

def main():
    #get the city population data

    df_city_p = pd.read_csv('./data/city_population.csv',delimiter=';'
                            ,encoding='latin1',header=None)
    
    #drop other columns, only take the 2023 data
    df_city_p = df_city_p.drop(columns=[0,2,3,4,5])

    #change column name
    col_name = {1:'city', 6:'population'}

    df_city_p.rename(columns=col_name,inplace=True)

    # change data type of population to int
    df_city_p['population'] = df_city_p['population'].astype('Int64')

    #Consider only the cities with population more than 100000
    df_city_p = df_city_p[df_city_p['population']>=100000]

    #extract only the city name

    df_city_p['city'] = df_city_p['city'].str.split(',',expand=True)[0]

    print(df_city_p.head(10))
    # print("Hello")

def get_city_from_df(str):
    str = str.split(',')[0].strip()

if __name__ == '__main__':
    main()