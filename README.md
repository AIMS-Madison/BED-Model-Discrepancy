# BED-Model-Discrepancy

This repository supports the paper *"Active Learning of Model Discrepancy with Bayesian Experimental Design"*.

## Quick Start

### Environment Requirements

- `jax`         0.4.20 ~ 0.4.28  
- `jaxlib`      0.4.18+cuda12.cudnn89 ~ 0.4.28+cuda12.cudnn89  
- `jax-cfd`     0.2.0  
- `flax`        0.8.2  
- `optax`       0.2.2 ~ 0.2.4  

⚠️ *Note: `optax` 0.2.4 is only compatible with `jax` 0.4.28. Please ensure version consistency.*

Experiments in the paper were originally run on **WSL with an RTX 4090**. The code is also known to **exceed memory limits** on a 12GB **4070 Super**. CPU execution is possible but may be slower.

---

### Contents

- `Structural error case.ipynb`:  
  Jupyter notebook to reproduce Figs. 7, 8, and 9 in the paper.

- `jax_cfd_test/`:  
  Necessary components for the custom PDE solver:
  - `my_equations.py`: core PDE-solving framework  
  - `my_forcing.py`: forcing term used in the PDE  
  - `my_funcutils.py`: utility functions for the AD-based solver

### Citation
If you use this code or find it helpful for your research, please cite:

```bibtex
@article{yang2025active,
  title={Active learning of model discrepancy with Bayesian experimental design},
  author={Yang, Huchen and Chen, Chuanqi and Wu, Jin-Long},
  journal={arXiv preprint arXiv:2502.05372},
  year={2025}
}
