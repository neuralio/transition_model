import pandas as pd

def load_pv_data(pv_suitability_file, pv_yield_file):
    """
    Load and preprocess PV suitability and PV profit data.
    """
    # PV Suitability
    PV_ = pd.read_csv(pv_suitability_file, sep=',')
    PV_ = PV_.rename(columns={"score": "pv_suitability"})
    PV_ = PV_[(PV_["year"] >= 2021) & (PV_["year"] <= 2030)]
    print("Loaded PV Suitability")

    # PV Yield (Profit)
    PV_Yield_ = pd.read_csv(pv_yield_file, sep=',')
    PV_Yield_ = PV_Yield_.rename(columns={'annual_electricity_savings_€': 'pv_profit'})
    print("Loaded PV Yield")

    return PV_, PV_Yield_

# def load_crop_data(cs_maize_file, cs_wheat_file, cy_maize_file, cy_wheat_file):
#     """
#     Load and preprocess crop suitability and profit data for maize and wheat.
#     """
#     # Crop Suitability
#     CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
#     CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
#     crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
#     crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
#     crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
#     print("Loaded Crop Suitability (Maize & Wheat)")

#     # Crop Yield for Maize
#     CY_Maize = pd.read_csv(cy_maize_file, sep=',')[['Dates', 'Profits']].copy()
#     CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
#     CY_Maize['year'] = CY_Maize['Dates'].dt.year
#     CY_Maize = CY_Maize.groupby('year').last().reset_index()
#     CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
#     print("Processed Crop Yield for Maize")

#     # Crop Yield for Wheat
#     CY_Wheat = pd.read_csv(cy_wheat_file, sep=',')[['Dates', 'Profits']].copy()
#     CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
#     CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
#     CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
#     CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
#     print("Processed Crop Yield for Wheat")

#     # Combine Crop Profit
#     crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
#     return crop_suitability, crop_profit
def load_crop_data(cs_maize_file, cs_wheat_file, cy_maize_file, cy_wheat_file):
    """
    Load and preprocess crop suitability and profit data for maize and wheat.
    """
    # Crop Suitability
    CS_Maize_ = pd.read_csv(cs_maize_file, sep=',')
    CS_Wheat_ = pd.read_csv(cs_wheat_file, sep=',')
    crop_suitability = pd.concat([CS_Maize_, CS_Wheat_], ignore_index=True)
    
    # Adjust filtering based on past/future data
    if "year" in crop_suitability.columns:
        crop_suitability = crop_suitability[(crop_suitability["year"] >= 2021) & (crop_suitability["year"] <= 2030)]
    crop_suitability = crop_suitability.rename(columns={"score": "crop_suitability"})
    print("Loaded Crop Suitability (Maize & Wheat)")

    # Crop Yield for Maize
    CY_Maize = pd.read_csv(cy_maize_file, sep=',')[['Dates', 'Profits']].copy()
    CY_Maize['Dates'] = pd.to_datetime(CY_Maize['Dates'])
    CY_Maize['year'] = CY_Maize['Dates'].dt.year
    CY_Maize = CY_Maize.groupby('year').last().reset_index()
    CY_Maize = CY_Maize.rename(columns={'Profits': 'crop_profit'})
    print("Processed Crop Yield for Maize")

    # Crop Yield for Wheat
    CY_Wheat = pd.read_csv(cy_wheat_file, sep=',')[['Dates', 'Profits']].copy()
    CY_Wheat['Dates'] = pd.to_datetime(CY_Wheat['Dates'])
    CY_Wheat['year'] = CY_Wheat['Dates'].dt.year
    CY_Wheat = CY_Wheat.groupby('year').last().reset_index()
    CY_Wheat = CY_Wheat.rename(columns={'Profits': 'crop_profit'})
    print("Processed Crop Yield for Wheat")

    # Combine Crop Profit
    crop_profit = pd.concat([CY_Maize, CY_Wheat], ignore_index=True)
    return crop_suitability, crop_profit


def merge_data(crop_suitability, PV_, PV_Yield_, crop_profit):
    """
    Merge all datasets into a single DataFrame.
    """
    # Merge datasets
    crop_profit = crop_suitability.merge(crop_profit, on="year", how="inner")[["year", "lat", "lon", "crop_profit"]]
    for df in [PV_, crop_suitability, PV_Yield_, crop_profit]:
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
    env_data = crop_suitability.merge(PV_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(PV_Yield_, on=["year", "lat", "lon"], how="inner")
    env_data = env_data.merge(crop_profit, on=["year", "lat", "lon"], how="inner")
    print("Merged environmental dataset")
    return env_data

def aggregate_data(env_data):
    """
    Aggregate data by unique lat/lon pairs.
    """
    aggregated_data = env_data.groupby(['lat', 'lon'], as_index=False).agg({
        'crop_suitability': 'mean',  # Average crop suitability over years
        'pv_suitability': 'mean',   # Average PV suitability over years
        'crop_profit': 'sum',       # Total crop profit over years
        'pv_profit': 'sum'          # Total PV profit over years
    })
    unique_lat_lon_count = aggregated_data[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"Number of unique lat/lon pairs: {unique_lat_lon_count}")
    return aggregated_data

