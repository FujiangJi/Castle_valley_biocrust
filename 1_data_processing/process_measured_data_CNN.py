import pandas as pd
import numpy as np

df_spectra = pd.read_csv(f"../1_data/ASD_Specra/ASD_All_Spectra_PlotLevel.csv")

df_cover = pd.read_csv(f"../1_data/USGS Cover Data/FractionaCover_BySpectra_2021.csv")
df_cover.loc[df_cover["Treatment"] == "C", "Treat"] = "CC"
df_cover.loc[df_cover["Treatment"] == "L", "Treat"] = "LC"
df_cover.loc[df_cover["Treatment"] == "LW", "Treat"] = "LW"
df_cover.loc[df_cover["Treatment"] == "W", "Treat"] = "CW"
df_cover['subplots'] = df_cover['Spectra'].str[1:]\

df_cover = df_cover[["Plot", "Block", "Treat", 'subplots',"Litter", "Rock", "Bare", "PlantBase", "Lichen", "Moss", "Dark", "Light"]]
df_cover["full"] = df_cover["Plot"].astype(str) + df_cover["Block"].astype(str) + "_" + df_cover["Treat"].astype(str) + "_" + df_cover["subplots"].astype(str)

df_cover = df_cover[["full", "Litter", "Dark", "Lichen", "Light", "Moss", "PlantBase"]]

df_cover.columns = "full", 'frac_Litter', 'frac_DkCy', 'frac_Lichen', 'frac_LtCy', 'frac_Moss', 'frac_Vegetation'
df_measure = pd.merge(df_spectra, df_cover, how = 'left', on = 'full')
df_measure.dropna(subset = ['frac_Litter', 'frac_DkCy', 'frac_Lichen', 'frac_LtCy', 'frac_Moss', 'frac_Vegetation'], how="all", inplace = True)
df_measure.reset_index(drop = True, inplace = True)
df_measure.to_csv("../1_data/Processed_data/measured_mixtures.csv", index=False)

eps = 1e-8
df_measure["brightness_index"] = np.sqrt((df_measure["560"]**2 + df_measure["650"]**2 + df_measure["850"]**2) / 3)
df_measure["NDVI"] = ((df_measure["800"] - df_measure["670"]) /(df_measure["800"] + df_measure["670"]+ eps))
df_measure["PRI"] = ((df_measure["531"] - df_measure["570"]) /(df_measure["531"] + df_measure["570"]+ eps))
df_measure["NDNI"] = ((np.log(df_measure["1680"]) - np.log(df_measure["1510"]+ eps)) /(np.log(df_measure["1680"]) + np.log(df_measure["1510"]+ eps)))
df_measure["NDWI"] = ((df_measure["860"] - df_measure["1240"]) / (df_measure["860"] + df_measure["1240"]+ eps))
df_measure["MCARI"] = (((df_measure["700"] - df_measure["670"]) - 0.2 * (df_measure["700"] - df_measure["550"])) *(df_measure["700"] / df_measure["670"]+ eps))
df_measure["soil_moisture"] = ((df_measure["1850"] - df_measure["1925"]) /(df_measure["1850"] + df_measure["1925"]+ eps))
df_measure.to_csv("../1_data/Processed_data/measured_mixtures_with_indices.csv", index=False)

df_measure_copy = df_measure.copy()
#**********************************************************************************************************#
target_cols = ["frac_Litter", "frac_DkCy", "frac_Lichen", "frac_LtCy", "frac_Moss", "frac_Vegetation"]
df_measure["plot_key"] = df_measure["full"].apply(lambda x: "_".join(x.split("_")[:2]))

df_spectra = df_measure.loc[:,"350":"2500"]
df_cover = df_measure[target_cols]

df_spectra = pd.concat([df_measure[["plot_key"]], df_spectra], axis = 1)
df_cover = pd.concat([df_measure[["plot_key"]], df_cover], axis = 1)

df_spectra = df_spectra.groupby("plot_key").mean().reset_index()
df_cover = df_cover.groupby("plot_key").sum()
df_cover = df_cover.div(df_cover.sum(axis=1), axis=0).reset_index()

df_measure = pd.merge(df_spectra, df_cover, how = 'left', on = 'plot_key')
df_measure.rename(columns={"plot_key": "full"}, inplace=True)
df_measure.reset_index(drop = True, inplace = True)
df_measure.to_csv("../1_data/Processed_data/plot_aggregated_measured_mixtures.csv", index=False)

#**********************************************************************************************************#
index_cols = ["brightness_index", "NDVI", "PRI", "NDNI", "NDWI", "MCARI", "soil_moisture"]
df_measure_copy["plot_key"] = df_measure["full"].apply(lambda x: "_".join(x.split("_")[:2]))

df_spectra = df_measure_copy.loc[:,"350":"2500"]
df_indices = df_measure_copy[index_cols]
df_spectra = pd.concat([df_spectra, df_indices], axis = 1)
df_cover = df_measure_copy[target_cols]

df_spectra = pd.concat([df_measure_copy[["plot_key"]], df_spectra], axis = 1)
df_cover = pd.concat([df_measure_copy[["plot_key"]], df_cover], axis = 1)

df_spectra = df_spectra.groupby("plot_key").mean().reset_index()
df_cover = df_cover.groupby("plot_key").sum()
df_cover = df_cover.div(df_cover.sum(axis=1), axis=0).reset_index()

df_measure = pd.merge(df_spectra, df_cover, how = 'left', on = 'plot_key')
df_measure.rename(columns={"plot_key": "full"}, inplace=True)
df_measure.reset_index(drop = True, inplace = True)
df_measure.to_csv("../1_data/Processed_data/plot_aggregated_measured_mixtures_with_indices.csv", index=False)