## Overview
We trained a machine learning model by using mixture density network (MDN) algorithm to predict the inner structure of rocky exoplants.

## Quick Start
### Step 1:
[Fork and clone](https://help.github.com/articles/fork-a-repo) a copy of the `Rocky_Exoplanets` repository on to your local machine.

### Step 2:
Download [`Anaconda`](https://www.anaconda.com/products/individual#Downloads) and install it on you machine.
Create a `conda` environment called `Rocky_Exoplanets` and install all the necessary dependencies:

    $ conda create -n Rocky_Exoplanets pip python jupyter
    
### Step 3:
Activate the `Rocky_Exoplanets` environment:

    $ conda activate Rocky_Exoplanets

### Step 4:
Change into your local copy of the `Rocky_Exoplanets` repo:

    $ cd /you own path/Rocky_Exoplanets

### Step 5:
Install the requirments for predicing in the current Conda environment:

    $ pip install -r requirements.txt

### Step 6:
Open `Jupyter Notebook` and load the file `MDN_prediction.ipynb`:

    $ jupyter notebook

At this point you are ready to start making prediction!
