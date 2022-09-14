# INPCA403
Incremental normalised principal component analysis (INPCA) for large datasets. The number 403 is a random number added
to make sure the package name is unique.

I developed this package for my own use, after attempting to do a PCA on a very large dataset for a Kaggle competition.
The dataset was too large to fit in memory for a normal PCA. For this end I found sklearns IPCA class, however I also
wanted to normalise the data (zero mean, unit variance), using multiple cores to speed up the normalisation.

# Installation

It is recommended to install this package in a virtual environment (not explained here).
Within your chosen python installation, install the requirements, and then this package with the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

# Usage

To do the INPCA, you have to create a Chunker that loads the data in chunks,
and an INPCA model that is fitted to the Chunker object. The Chunker has a method `get_chunk(i)` that returns chunk i
of the data. See example script in folder `scripts`.

More Chunker classes for loading other types of data than HDF can be created following the same pattern as HDFChunker, as long
as they have a method for getting a chunk of data, and returning the total number of chunks (see Chunker class in
inpca403/inpca403.py).

