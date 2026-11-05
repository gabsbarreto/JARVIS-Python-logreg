# Data and provenance

The pickle files are gzip-compressed copies of prepared local JARVIS datasets. Unused SVD columns and the cocaine dataset's unused PCA columns have been removed; all other fields and record order are preserved. They include publication text, screening and full-text labels, PICO responses, and sometimes author details or reviewer notes. The scripts rebuild TF-IDF directly from the abstracts.

The two `antiacoag` files and the similarly named `anticoag` files are distinct prepared versions. The project preserves both names and does not assume their inputs are identical.

The provenance and redistribution terms for each prepared dataset and `smart.txt` have not been established here. The repository's MIT software licence does not establish rights to redistribute third-party datasets, publication abstracts, notes, or the stopword file. Verify ownership, consent where relevant, and applicable terms before making those files public.

The files were produced with a pandas 3.x environment. Pickle compatibility can depend on the pandas and Python versions used. **Only open pickle files from a trusted source:** loading a malicious pickle can execute code.

From the project root, check the copied files with `shasum -a 256 -c data/SHA256SUMS` on macOS, or `sha256sum -c data/SHA256SUMS` on Linux.
