# pydlem
 Python implementation of the daily version of the Lake Evaporation Model (LEM; Zhao et al. 2024).

 ## Installation
 ### Option 1:
 To install the current stable version, install from git using pip
 pip install git+https://github.com/MTDNRC-WRD/pydlem@main
 This has not been tested.
 
 ### Option 2:
 Clone the repo to a local directory and use os.chdir() at the beginning of a notebook or script to point to the cloned repo. This was tested and works.
 You will need to set up the environment correctly for this option to run.
 #### pydlem environment:
 The repo contains a pydlem.yml environment file, you can attempt to create the environment from the .yml using conda (this seems to work inconsistently).

 The other option is manually setting up the environment as follows:
 
 conda create -n pydlem python=3.9
 
 conda activate pydlem
 
 pip install git+https://github.com/MTDNRC-WRD/chmdata@main
 
 conda install -c conda-forge py3dep
 
 conda install ipykernel tqdm tomli  (ipykernel so that jupyter notebook will recognize the environment as a kernel)

 conda install xvec -c conda-forge
