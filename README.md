# Byzantine-Robust Loopless Stochastic Variance-Reduced Gradient

This code comes jointly with reference
> [1] Nikita Fedin, Eduard Gorbunov. "Byzantine-Robust Loopless Stochastic Variance-Reduced Gradient"
Date: March 2023

## Requirements

**Packages.** No special packages beyond standard ones are required.

**Folders.** Before running the code, one needs to create three folders in the directory with the code: "datasets", "dump", "plot". In the directory "datasets", 
one should put datasets "a9a", "w8a", "phishing", and "mushrooms" from LIBSVM library https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.

## Files

- "algs.py" contains the implementation of the algorithms
- "functions.py" contains the main functions for computing gradients, loss, and estimators
- "utils.py" contains functions to work with data and for visualization of the plots
- jyputer notebooks contain the code with experiments
