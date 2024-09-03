# Messina_2024
Code and data from Messina et al. 2024

• Hi-M data were acquired using qudi-HiM (Barho et al. 2022). The current version of qudi-HiM is found at https://github.com/NollmannLab/qudi-HiM, and an archived version at https://zenodo.org/record/6379944 (DOI: 10.5281/zenodo.6379944).

• Hi-M data were analyzed using pyHiM release 0.7, available at https://github.com/marcnol/pyHiM (Devos et al. 2024). 

• Hi-M traces as well as matrices are accessible at OSF repository: https://osf.io/27dnx/?view_only=71678f1fcd924f7db81384f529c041b8. 

• Script_figures contain all python scripts used to generate all figure pannels from Messina et al. 2024. Sources data used in the scripts are accessible at OSF repository: https://osf.io/27dnx/?view_only=71678f1fcd924f7db81384f529c041b8. 

• Script_analysis contains the scripts used for the analysis:

	- Normalization : scripts used to normalize the PWD matrix from multiple experiments
	- DAPI_registration : scripts used to register the DAPI images from DNA & RNA-FISH experiments
	- Cell-type_trace_selection : scripts used to assign trace to specific cell types (exocrine, beta cells, etc.)
	- Topic_analysis : scripts used to perform LDA decomposition on HiM data and analyze the data

