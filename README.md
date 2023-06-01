HyperSLICE: HyperBand optimised Spiral for Low-latency Interactive Cardiac Examination
======================================================================================

Dr. Olivier Jaubert, Dr. Javier Montalt-Tordera, Dr. Daniel Knight, Pr.
Simon Arridge, Dr. Jennifer Steeden, Pr. Vivek Muthurangu

Installation
============

All required packages can be installed in a virtual environment:

``` {.console}
$ conda env create --name hyperslice --file environment.yml
```

Note that only Linux is supported.

Training and testing
====================

Once the environment created, activate using:

``` {.console}
$ conda activate hyperslice.
```

You can then run:

-   the python file example.py file from project directory.

``` {.console}
$ python example.py
```

-   or the ipython notebook example.ipynb (to see intermediate results)

Results are saved in ./Training\_folder (as in the already trained
exemple model ./Training\_folder/Exemple\_Trained\_FastDVDnet)

Logs can be visualised using tensorboard:

``` {.console}
$ tensorboard --logdir path_to_directory
```

Acknowledgments
===============

Code relies heavily on TensorFlow \[1\] and TensorFlow\_MRI \[2\]:

\[1\] Abadi M, Barham P, Chen J, et al. TensorFlow: A System for
Large-Scale Machine Learning TensorFlow: A system for large-scale
machine learning. Proceedings of the 12th USENIX Symposium on Operating
Systems Design and Implementation 2016:265--283.

\[2\] Montalt Tordera J. TensorFlow MRI. 2022 doi:
10.5281/ZENODO.7120930.
