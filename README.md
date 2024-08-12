# Measure-Theoretic Time-Delay Embedding

This repository contains Python code which can be used to learn the Takens' embedding full state reconstruction map as a pushforward between probability measures. We include code which can be used to reproduce our experiments on low-dimensional test synthetic dynamical systems. The NOAA SST and ERA5 wind speed datasets can be made available upon request. 

- `generate_data.py`: Generates data from the dynamical systems we wish to learn the reconstruction map for.
- `generate_embedding.py`: Estimates the embedding parameters for a scalar observable of the full state, forms the time-delay coordinates, and partitions the data into training and testing phases.
- `generate_patches_sparse.py`: Deploys a constrained k-means clustering routine to build the measures used to perform the full state reconstruction.
- `train_measures.py`: Learns the reconstruction map from data by minimizing the MMD between the pushforward of measures in delay coordinates with the corresponding measures in reconstruction space.
- `train_pointwise.py`: Learns the reconstruction map by minimizing the MSE.
- `plot_results_lorenz.py`: Visualizes the reconstruction results for the two approaches and report the error. 

The code in this repository can be used to reproduce the following comparison for reconstructing the Lorenz-63 system based on noisy partial observations. 
![lorenz](https://github.com/user-attachments/assets/51add5b4-b863-4bc3-a724-596fb2b27306)

The following visualizes the discrete measures in time-delay coordinates we use to reconstruct the chaotic Rossler system.

![Slide2 (1)](https://github.com/user-attachments/assets/fd1113d4-013f-4a6b-ba5e-be8714abfc45)

The following videos show visual comparisons for the measure-based and pointwise-based reconstructions of the NOAA SST and ERA5 wind speed datasets. For more details on the experimental setup, please refer to the paper. 

- **NOAA Reconstruction:** https://drive.google.com/file/d/1_F-SimPW-PGcOFTTOYj-9Ew4xP4vWtZs/view?usp=sharing
- **ERA5 Reconstruction:** https://drive.google.com/file/d/1Ncdmlhj2r8mMwXbCJPAshV67R5ElfpBg/view?usp=sharing

