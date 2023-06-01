import os
import logging
from preprocess.data_loader import CsvLoader

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    train_test = 'test'

    loader = CsvLoader('/home/yichaoli8/fed_csis23')
    write_dir = os.path.join('/home/yichaoli8/fed_csis23/shards', 'test')
    n_samples = len(loader.data_files[train_test])
    for i, lower in enumerate(range(0, n_samples, 2000)):
        # if i < 6:
        #     continue
        upper = lower + 2000
        df = loader.load_row_range(train_test, slice(lower, upper))
        fname = 'shard%i.parquet' % i
        write_path = os.path.join(write_dir, fname)
        logging.info('writing parquet %s' % write_path)
        df.to_parquet(write_path, engine='pyarrow', compression='gzip', index=False)


    loader.pool.close()
    loader.pool.join()
    # make local train set
    # loader = CsvLoader('/home/yichaoli8/fed_csis23')
    # write_path = '/home/yichaoli8/fed_csis23/shards/train_local/shard0.parquet'
    # df = loader.load_sample_csv(1000, 522)
    # df.to_parquet(write_path, engine='pyarrow', compression='gzip', index=False)
