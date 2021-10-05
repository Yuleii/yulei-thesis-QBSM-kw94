# Quantile-based Sensitivity Analysis on Structural Behavioral models


The goal of this project was to apply
Quantile-based sensitivity analysis to structural econometric models. Specifically, [Keane and Wolpin (1994)](https://www.jstor.org/stable/2109768) model is used as an application.


## Repository Structure

The repository is organized as follows:
* **data**: contains the data used during the execution of the program.
* **figures**: contains the output figures genetated by plot scripts.
* **functions**: contains the code for the project.
	* **sampling.py** contains functions to generate the input data.
	* **qoi.py** contains functions to compute the quantity of interest.
  	* **QBSM.py** contains functions to calculate the [quantile-based sensitivity measures](https://www.sciencedirect.com/science/article/abs/pii/S0951832016304574) using Monte Carlo methods.
	* **sobol_indices.py** contains functions to calculated sobol indices for Keane and Wolpin (1994) model. It is based on the [SALib](https://salib.readthedocs.io/en/latest/"SALib") library.
   	* **plot** contains scripts for reproducing the figures in the thesis.
* **tex**: contains the main thesis document.
* **thesis-replication.ipynb** contains the full execution pipeline for the project.






