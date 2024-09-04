# Messina_2024
Code and data from Messina et al. 2024

Hi-M data were acquired using qudi-HiM (Barho et al. 2022). The current version of qudi-HiM is found at https://github.com/NollmannLab/qudi-HiM, and an archived version at https://zenodo.org/record/6379944 (DOI: 10.5281/zenodo.6379944).

Hi-M data were analyzed using pyHiM release 0.7, available at https://github.com/marcnol/pyHiM (Devos et al. 2024). 

Hi-M traces as well as matrices are accessible at OSF repository: https://osf.io/27dnx/?view_only=71678f1fcd924f7db81384f529c041b8. 

The `Script_figures` folder contains all python scripts used to generate all figure panels from Messina et al. 2024.

The `Script_analysis` folder contains the scripts used for the analysis of:

	- `Normalization`: scripts used to normalize the PWD matrix from multiple experiments
	- `DAPI_registration`: scripts used to register the DAPI images from DNA & RNA-FISH experiments
	- `Cell-type_trace_selection`: scripts used to assign trace to specific cell types (exocrine, beta cells, etc.)
	- `3DTopics`: scripts used to perform LDA decomposition on HiM data and analyze the data
