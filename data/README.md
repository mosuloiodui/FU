# Data

The CSV files are not versioned (several hundred MB).

Source: **UNSW-NB15** (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`),
available on the [official UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

Put the two raw files in this folder, then run [`notebooks/prepro.ipynb`](../notebooks/prepro.ipynb).
It writes the non-IID partition used by the federated system:

| Machine | Train file | Local test file |
|---|---|---|
| VM 1 (`client_0`, Client 1) | `client_0_train.csv` | `test_0.csv` |
| VM 2 (`client_1`, Client 2, holds `Generic`) | `client_1_train.csv` | `test_1.csv` |
| VM 3 (`client_2`, Client 3) | `client_2_train.csv` | `test_2.csv` |
| Any (neutral evaluation) | | `global_test.csv` |

Each VM only needs its own two files. The server needs no data.
