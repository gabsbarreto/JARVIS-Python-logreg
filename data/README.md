# Data

This directory contains seven prepared, gzip-compressed datasets and the `smart.txt` stopword list used by the TF-IDF pipeline. Each dataset includes abstracts, screening and full-text decisions, R/P/I/C/O/S responses and the total score used for initial sampling.

| Dataset | Records |
| --- | ---: |
| Hypertension | 7,452 |
| Anticoag | 1,632 |
| Anticoag2 | 2,420 |
| PO-UR | 2,045 |
| Cocaine | 1,901 |
| CYP1A2 | 807 |
| Portion size | 46,494 |

The files were prepared using pandas 3.x. Python and pandas versions can affect pickle compatibility. **Only load pickle files from trusted sources:** a malicious pickle can execute code.

The datasets include publication text and may contain author details or reviewer notes. Their provenance and reuse terms, and those of `smart.txt`, have not been established in this project. The MIT software licence does not establish rights to reuse or redistribute third-party content; check the applicable permissions before doing so.

From the project root, verify the files with `shasum -a 256 -c data/SHA256SUMS` on macOS, or `sha256sum -c data/SHA256SUMS` on Linux.
