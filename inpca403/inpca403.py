import pickle, os
from abc import ABC

import numpy as np
from sklearn.decomposition import IncrementalPCA
from joblib import Parallel, delayed
import pandas as pd

class Chunker(ABC):
    def get_chunk(self, i):
        return

    def get_num_chunks(self):
        return

class HDFChunker(Chunker):
    def __init__(self, hdf_filepath, chunk_size=500):
        self.chunk_size = chunk_size
        self.hdf_filepath = hdf_filepath
        # find dimensions of data
        store = pd.HDFStore(hdf_filepath)
        df_key = store.keys()[0]
        self.nrows, self.ncols = store.get_storer(df_key).shape
        store.close()

        self.num_chunks = self.nrows // self.chunk_size


    def get_chunk(self, i):
        if(i <0 or i>=self.num_chunks):
            raise ValueError("Index out of range")
        start = i*self.chunk_size
        stop = start + self.chunk_size
        df = pd.read_hdf(self.hdf_filepath, start=start, stop=stop)
        return df

    def chunk_range(self):
        return range(self.num_chunks)

    def get_num_chunks(self):
        return self.num_chunks


class INPCA():

    def __init__(self, n_components = 20):

        self.n_components = n_components
        self.averages = []
        self.stds = []
        return


    def _calc_averages(self, hdf_chunker):
        def calc_sums(chunk_i):
            df = hdf_chunker.get_chunk(chunk_i)
            print(f"{chunk_i} of {hdf_chunker.get_num_chunks()-1}")
            return df.mean(axis=0)

        results = Parallel(n_jobs=8)(delayed(calc_sums)(i) for i in hdf_chunker.chunk_range())
        self.averages = np.mean(results, axis=0)
        return self

    def _calc_stds(self, hdf_chunker):
        if len(self.averages) != hdf_chunker.ncols:
            print("Warning, number of averages not the same as number of columns. Re-calculating averages...")
            self._calc_averages(hdf_chunker)
            print("Averages calculated!")

        def calc_sums_of_squares(chunk_i):
            df = hdf_chunker.get_chunk(chunk_i)
            print(f"{chunk_i} of {hdf_chunker.get_num_chunks()-1}")
            return ((df - self.averages)**2).mean(axis=0)

        results = Parallel(n_jobs=8)(delayed(calc_sums_of_squares)(i) for i in hdf_chunker.chunk_range())
        stds = np.sqrt(np.sum(results, axis=0))

        # identify columns with zero variance
        print(f"Number of columns with zero standard deviation: {(stds == 0).sum()}")
        zero_variance = (stds == 0)

        stds[zero_variance] = 1 # set to 1 to avoid division by zero

        self.stds = stds
        return self

    def fit_inpca(self, hdf_chunker):
        if len(self.stds) != hdf_chunker.ncols:
            print("Warning, number of stds not the same as number of columns. Re-calculating stds...")
            self._calc_stds(hdf_chunker)
            print("stds calculated!")

        self.ipca = IncrementalPCA(n_components=self.n_components)

        for i in hdf_chunker.chunk_range():
            df = hdf_chunker.get_chunk(i)
            norm = (df-self.averages)/self.stds

            # fit PCA
            self.ipca.partial_fit(norm)
            print(f"PCA: {i} of {hdf_chunker.get_num_chunks()-1} done.")

        print(f"Total amount of explained variance: {np.sum(self.ipca.explained_variance_ratio_)}")
        return self

    def transform_and_save_to_csv(self, hdf_chunker, save_path):

        # delete old file if it exists
        if os.path.exists(save_path):
            print(f"{save_path} already exists. Deleting...")
            os.remove(save_path)

        for i in hdf_chunker.chunk_range():
            df = hdf_chunker.get_chunk(i)
            norm = (df-self.averages)/self.stds

            # transform with the PCA
            norm_transformed = self.ipca.transform(norm)

            pd.DataFrame(norm_transformed).to_csv(save_path, mode='a', index=False, header=False)

            print(f"Transform: {i} of {hdf_chunker.get_num_chunks()-1} done.")
        return self

    def to_pickle(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        return self
