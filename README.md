This folder consists of codes, functions and datasets (placeholder) to run flood risk/resilience analysis for the US Naval Facility in Annapolis, MD. The codes utilize some functionalities of INCORE, but mostly contain scripts and functions that run all the analysis outside INCORE. An authentication is required to connect to the INCORE account.

As a first step, plese run the Notebook 'create_power_network.ipynb' to read the locations of the power plants, substations, generate power lines joining these faiclities, as well as read building locations and create local power distribution lines feeding the buildings. The input data required for running the main script are all generated and saved inside the folder titled 'input_data'. The transportation network data is already preloaded in this folder, along with the prerun outputs of 'create_power_network.ipynb'.

The Notebook titled 'NAVFAC_analysis_new.ipynb' performs the main analysis and generates the plots. All supporting functions are contained in the python script 'utility_functions.py'.

NOTE: In order to make any changes to the input datasets, please update the exccel file 'EPN_facilities.xlsx' wihout changing any of the column names and structure.
