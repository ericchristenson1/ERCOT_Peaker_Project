import pandas as pd

# Combine weather data from multiple CSV files into a single DataFrame
def combine_weather_data(file_list):
    combined_df = pd.DataFrame()
    
    for file in file_list:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

if __name__ == "__main__":
    files = [
        "data/2021_DFW.csv",
        "data/2022_DFW.csv",
        "data/2023_DFW.csv",
        "data/2024_DFW.csv",
    ]
    
    combined_weather_df = combine_weather_data(files)
    print(combined_weather_df.head())

    # Save the csv to data/combined_DFW_weather.csv
    combined_weather_df.to_csv("data/combined_DFW_weather.csv", index=False)