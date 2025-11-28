import pandas as pd

# Combine weather data from multiple CSV files into a single DataFrame
def combine_weather_data(file_list):
    combined_df = pd.DataFrame()
    
    for file in file_list:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

def clean_weather_data(df):
    # ---------------------------
    #  Helper: split "value,QC"
    # ---------------------------
    def split_grouped(series):
        s = series.astype(str).str.split(",", expand=True)
        s.columns = ["value", "qc"]
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s["qc"] = pd.to_numeric(s["qc"], errors="coerce")
        return s

    # ---------------------------------
    # 2. Temperature (°C) and Dew Point
    # ---------------------------------
    if "TMP" in df.columns:
        temp = split_grouped(df["TMP"])
        df["TEMP_C"] = temp["value"].replace(9999, pd.NA) / 10
        df["TEMP_qc"] = temp["qc"]

    if "DEW" in df.columns:
        dew = split_grouped(df["DEW"])
        df["DEW_C"] = dew["value"].replace(9999, pd.NA) / 10
        df["DEW_qc"] = dew["qc"]

    # --------------------------------
    # 3. Pressures (hPa) — SLP & STP
    # --------------------------------
    if "SLP" in df.columns:
        slp = split_grouped(df["SLP"])
        df["SLP_hPa"] = slp["value"].replace(99999, pd.NA) / 10
        df["SLP_qc"]  = slp["qc"]

    # -------------------------------------
    # 5. Wind — expanded format:
    #    dir,dir_qc,type,speed,speed_qc
    #    e.g. "030,1,N,0093,1"
    # -------------------------------------
    if "WND" in df.columns:
        wnd_split = df["WND"].astype(str).str.split(",", expand=True)
        wnd_split.columns = ["dir", "dir_qc", "type", "spd", "spd_qc"]

        df["WIND_DIR_deg"] = pd.to_numeric(wnd_split["dir"], errors="coerce").replace(999, pd.NA)
        df["WIND_DIR_qc"]  = pd.to_numeric(wnd_split["dir_qc"], errors="coerce")
        df["WIND_TYPE"]    = wnd_split["type"]
        df["WIND_SPD_ms"]  = pd.to_numeric(wnd_split["spd"], errors="coerce").replace(9999, pd.NA) / 10
        df["WIND_SPD_qc"]  = pd.to_numeric(wnd_split["spd_qc"], errors="coerce")



    # ----------------------------------------------------
    # 7. Convert existing ISO datetime column → datetime
    #    e.g. "2021-01-01T00:00:00"
    # ----------------------------------------------------
    df["datetime"] = pd.to_datetime(df["DATE"])


    # ----------------------------------------------------
    # 8. Sort and inspect
    # ----------------------------------------------------
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    files = [
        "data/2025_DFW.csv",
    ]
    
    combined_weather_df = combine_weather_data(files)
    print(combined_weather_df.head())

    # Save the csv to data/combined_DFW_weather.csv
    combined_weather_df.to_csv("data/2025_DFW_weather.csv", index=False)

    # Clean the combined data
    cleaned_df = clean_weather_data(combined_weather_df)
    print(cleaned_df.head())

    # Save the cleaned data to clean_data/cleaned_DFW_weather.csv
    cleaned_df.to_csv("clean_data/cleaned_DFW_weather_2025.csv", index=False)