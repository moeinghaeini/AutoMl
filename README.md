[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/u_bn3D-w)
# Grey Box Optimization

## Assignment Instructions / TODOs

As you learned in the last lecture we can use Grey-Box Optimization to make the hyperparameter search more efficient.

Please refer to the [PDF](https://drive.google.com/file/d/1KLVeE9z2sy_Z4WoCPAdCdRbn3huN2B_e/view?usp=sharing) for full assignment instructions. 

## Installation

### Data preparation

For this exercise you will need to get the data from [1]. To download the datasets for the FC-Net benchmark you can just run `make download` or manually:

```bash
wget -P data http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf data/fcnet_tabular_benchmarks.tar.gz --directory data --strip-components=1
```

If `make download` or the above commands don't work for you, you can manually download the data and unpack it so that you have the following structure.

```
data
├── fcnet_naval_propulsion_data.hdf5
├── fcnet_parkinsons_telemonitoring_data.hdf5
├── fcnet_protein_structure_data.hdf5
├── fcnet_slice_localization_data.hdf5
└── fcnet_tabular_benchmarks.tar.gz
```

### Dependencies

`pip install -r requirements.txt`

### Running

Example run:
```python
python -m src.hyperband
```

### Feedback
Please give us feedback by filling out feedback.md file.

### Grading
The grading takes place in the following manner: 
- Autograding via Github Actions (This gives a brief idea of how well your solution performed).
- Respective Teaching Staff evaluates the solutions, and provide necessary feedback on the submitted solutions.

The final assignment grade would be the one decided after evaluation by the teaching staff associated with the exercises.
Overall, the assignments focus on comprehensive understanding of the lecture. 

#### Extra
These assignments were tested with `Python 3.10` and should work for any greater or equal version. You can check your python version by using `Python -V`.
It is highly advised to use some form of virtual environment when working on these assignments, to prevent conflicts between packages for different projects on your computer.
We recommend using Conda for this, but please feel free to use any method that works for you.

We also provide a `Makefile` which has some handy commands for you if you are using the commandline while doing this assignment, run `make help` to find them!

To run a `Makefile` on Windows, among other solutions, you can install `make` via Chocolatey as shown [here](https://chocolatey.org/install).


#### References

    [1] Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization
        A. Klein and F. Hutter
        arXiv:1905.04970 [cs.LG]

#### Deadline
The assignment is due on **CEST 23:59 of May 15, 2025**.

Incase of any issues, please reach out to us via discord classroom forum using the tag "Grey Box".
