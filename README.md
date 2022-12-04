# INPCA403
Incremental normalised principal component analysis (INPCA), for doing PCAs on large datasets that cannot be fit into memory. It is based on the scikit-learn
ICPA class, and currently only HDF files can be used with the package. The number 403 is a random number added
to make sure the package name is unique.

I developed this package for my own use, after attempting to do a PCA on a very large dataset for a Kaggle competition.
The dataset was too large to fit in memory for a normal PCA. I found the sklearns IPCA class, however it was not able to
 normalise the data (zero mean, unit variance), using multiple cores to speed up the normalisation.

## Installation

Install the requirements, and then this package with the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

# Usage

To do an INPCA, you have to create a Chunker that loads the data in chunks,
and an INPCA model that is fitted to the Chunker object. The Chunker has a method `get_chunk(i)` that returns chunk i
of the data. See example script in folder `scripts`.

More Chunker classes for loading other types of data than HDF can be created following the same pattern as HDFChunker,
as long as they have a method for getting a chunk of data, returning the total number of chunks, and a method for
returning all chunk indices (see Chunker class in inpca403/inpca403.py).

