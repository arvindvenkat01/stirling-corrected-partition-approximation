# Partition Approximation via the Cube Root of Binomial-Partition Ratios: A First-Principles Derivation with Stirling-Based Correction

This repository contains the Python code and results for the paper:

**Partition Approximation via the Cube Root of Binomial-Partition Ratios: A First-Principles Derivation with Stirling-Based Correction**

### Pre-print (Zenodo) : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17110539.svg)](https://doi.org/10.5281/zenodo.17110539)
* **DOI** - 10.5281/zenodo.17110539
* **URL** - https://doi.org/10.5281/zenodo.17110539


## Abstract
The paper introduces a novel, high-accuracy formula for approximating the integer partition function, p(n). 
The method is based on a first-principles derivation of the asymptotic behavior of the cube root of the ratio between the central binomial coefficient and p(n).

This base formula is enhanced with a theoretically-motivated correction factor, α(n)=3+1/(120n), which is derived from the structure of Stirling's approximation. The resulting formula is numerically stable and demonstrates superior accuracy to the classic Hardy-Ramanujan approximation across a wide range of practical values.

This script can be used to reproduce the error analysis and performance comparison plots presented in the paper.

## Features
The code includes:
- Exact partition function calculations via Euler's recurrence for validation
- Implementation of the cube root combination formula
- Application of the empirically optimized Stirling-based correction factor
- Benchmark results comparing performance against classical Hardy-Ramanujan asymptotics

## Repository Contents
* `partition_forms_analyzer.py` : Core Python script implementing the Stirling-corrected partition function approximation.
* `partition_error_summary_adaptive.csv` : Error summary table result output of the python script.
* `fig_partition_error_comparison.png` : Plot comparing the relative approximation errors (%) of the Stirling-corrected cube root partition function approximation against the fixed-factor theoretical formula and the Hardy-Ramanujan asymptotic.
* `README.md` : Documentation describing the project, installation, usage, and results.
* `requirements.txt` : List of Python dependencies required to run the code.

## Requirements
- Python 3.8+
- numpy, scipy, mpmath (for high-precision computations), matplotlib (optional for plotting)

## Installation

1.  **Clone the repository:**
  ```
git clone https://github.com/yourusername/stirling-corrected-partition-approximation.git
cd stirling-corrected-partition-approximation
```

2.  **Install the required dependency:**
```
pip install -r requirements.txt
```
    
### Example Output

```
============================================================
THEORETICAL HYBRID FORMS: EXTENDED COMPARISON
============================================================
Pre-computing partition values up to n=80000...
Pre-computation done in 4.40 seconds.

================================================================================
STARTING EXPERIMENT
Experiment finished.

Experiment completed in 0.00 seconds

Error Summary Table:
      Range             Formula Mean Error (%) Median Error (%) Max Error (%) Min Error (%)
   100-1000 Theoretical (Fixed)       2.190325         1.874146      4.310259      1.389893
   100-1000    Adaptive Scaling       2.026640         1.706663      4.172208      1.216216
   100-1000     Hardy-Ramanujan       2.265763         1.920853      4.571356      1.415244
 1001-10000 Theoretical (Fixed)       0.623758         0.570314      0.985293      0.442108
 1001-10000    Adaptive Scaling       0.439967         0.385813      0.806330      0.255870
 1001-10000     Hardy-Ramanujan       0.629155         0.574505      0.997917      0.444620
10001-80000 Theoretical (Fixed)       0.229213         0.207567      0.421586      0.156579
10001-80000    Adaptive Scaling       0.053953         0.029399      0.235068      0.000850
10001-80000     Hardy-Ramanujan       0.229942         0.208118      0.423869      0.156892

Saved 'partition_error_summary_adaptive.csv'
```


## Citation

If you use this work, please cite the paper using the Zenodo archive.

@misc{naladiga_venkat_2025_17110539,
  author       = {Naladiga Venkat, Arvind},
  title        = {Partition Approximation via the Cube Root of
                   Binomial-Partition Ratios: A First-Principles
                   Derivation with Stirling-Based Correction
                  },
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17110539},
  url          = {https://doi.org/10.5281/zenodo.17110539},
}

---

## License

The content of this repository is dual-licensed:

- **MIT License** for `partition_forms_analyzer.py` See the [LICENSE](LICENSE) file for details.
- **CC BY 4.0** (Creative Commons Attribution 4.0 International) for all other content (results.txt, README, etc.)



## Author

- **Arvind N. Venkat** - [arvind.venkat01@gmail.com](mailto:arvind.venkat01@gmail.com)
