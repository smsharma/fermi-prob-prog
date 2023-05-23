# Differentiable probabilistic programming for the GCE

- [X] Add ability to include multiple bulge models together with a simplex prior
- [X] Add ability to include multiple diffuse models together with a simplex prior
- [X] Sigmoid-bound problematic parameters to hedge against NaN? Update: Seems to work better with ELU -> Tanh nonlinearity
- [ ] See if n_exp > 1 works with more memory
- [X] Clean up pipeline for NeuTra -> MCMC
- [ ] Implement a GP template?
- [ ] Try running full pipeline (up to MCMC, potentially just SVI if including Ylm or GPs)

## Using Colaboratory

- Copy content of this repository to Google Drive. You can do this by first cloning the repository then uploading/syncing to Google Drive.
- Enable Google Colaboratory on Google Drive.
- Open notebooks with Colaboratory. Enable GPU in `Edit > Notebook settings` before running the notebook.
- Run the notebook including all the blocks that install required packages. Allow notebook's access to Google Drive when necessary.

------
## Authors
Siddharth Mishra-Sharma, Tracy Slatyer, Yitian Sun, and Yuqing Wu