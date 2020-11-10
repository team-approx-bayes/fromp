# FROMP

Contains code for the NeurIPS 2020 paper by Pan et al., "[Continual Deep Learning by Functional Regularisation of Memorable Past](https://arxiv.org/abs/2004.14070)".

## Continual learning with functional regularisation

FROMP performs continual continual by functionally regularising on a few memorable past datapoints, in order to avoid forgetting past information.

FROMP's functional regularisation is implemented in ``opt_fromp.py``. This is a PyTorch optimiser (built upon PyTorch's Adam optimiser), with FROMP's relevant additions. The important lines are in the ``step()`` function.

The provided scripts replicate FROMP's results from the paper. Running the code as in the Table below yields the reported average accuracy.
The files will run each experiment once. Change the ``num_runs`` variable to obtain mean and standard deviation over many runs (as reported in the paper).

| Benchmark | File | Average accuracy |
|---        |---   |---               |
| Split MNIST | ``main_splitmnist.py`` | 99.3% |
| Permuted MNIST | ``main_permutedmnist.py`` | 94.8% |
| Split CIFAR | ``main_cifar.py`` | 76.2% |
| Toy dataset | ``main_toydata.py`` | (Visualisation) |

### Further details

The code was run with ``Python 3.7``, ``PyTorch v1.2``. For the full environment, see ``requirements.txt``.

Hyperparameters (reported in Appendix F of the paper) are set in the ``main_*.py`` files. More detailed code is in the corresponding ``train_*.py`` files.

This code was written by Siddharth Swaroop and Pingbo Pan. Please raise issues here via github, or contact [Siddharth](ss2163@cam.ac.uk).

## Citation

```
@article{pan2020continual,
  title = {Continual Deep Learning by Functional Regularisation of Memorable Past},
  author = {Pan, Pingbo and Swaroop, Siddharth and Immer, Alexander and Eschenhagen, Runa and Turner, Richard E and Khan, Mohammad Emtiyaz},
  journal = {Advances in neural information processing systems},
  year = {2020}
}
```
