import inpca403.inpca403 as inpca403

hdf_file = "my_hdf_file.hd5"
# create the chunker that loads the data in chunks
train_hdf_chunker = inpca403.HDFChunker(hdf_file, chunk_size=100)

# create and fit the INPCA
mult_inpca = inpca403.INPCA(n_components=100).fit_inpca(train_hdf_chunker)

# save the trained INPCA object to disk
mult_inpca.to_pickle("saved_model.pkl")

# transform the data and save as csv
inpca.transform_and_save_to_csv(train_hdf_chunker, "transformed.csv")
