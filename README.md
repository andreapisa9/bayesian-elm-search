# Bayesian ELM Search
---
## License
This work is licensed under Creative Commons BY-NC-ND 4.0. Please **read the license text carefully** before handling the code.

## Attribution
If you choose to utilize our work for your research, please cite the following in your work:
```bibtex
@article{pau2024towards,
  title={Towards Full Forward On-Tiny-Device Learning: A Guided Search for a Randomly Initialized Neural Network},
  author={Pau, Danilo and Pisani, Andrea and Candelieri, Antonio},
  journal={Algorithms},
  volume={17},
  number={1},
  pages={22},
  year={2024},
  publisher={MDPI}
}
```
---
## How to Use
### Install the Required Libraries
The experiments were run on a `conda` environment. To reproduce them, please install `anaconda` or `miniconda` and create an empty environment using the following command:
```bash
conda create -n bayesian-elm-search python=3.9.12
```
After that, please activate the environment using the following command:
```bash
conda activate bayesian-elm-search
```
Finally, move to the project folder named `src` and install the required libraries using `pip` with the following command:
```bash
pip install -r requirements.txt
```
### Download the CIFAR-10 Dataset
Please download in a folder named 'CIFAR-10' the dataset from its [original source](https://www.cs.toronto.edu/~kriz/cifar.html). Be careful to download the **python version** (161 MB). In the same folder, download the Python script `perf_sample_idxs.py` from the [MLCommons Tiny repo](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification).

### Run the Experiments
Experiments are run using the following Python files:
- `FE_bayes_GP.py`, which uses Type 1 neural topology and Gaussian Processes as surrogate model;
- `FE_bayes_RF.py`, which uses Type 1 neural topology and Random Forests as surrogate model;
- `FE_bayes_RF_newtopology.py`, which uses Type 2 neural topology and Random Forests as surrogate model.

Within the Python scripts, please set the WORKING_DS string to be equal to the name of the dataset you may want to experiment with (i.e. 'MNIST' or 'CIFAR-10'). After that, launch the scripts one at a time using the following command:
```bash
python3 SCRIPTNAME.py > OUTPUT_SCRIPTNAME.txt
```

The results will be incrementally written on text files within the same folder. Please note that `FE_bayes_GP.py` automatically saves the results within a JSON file called `logs.log.json`, while the other two print the results on standard output.
