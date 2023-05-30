import os.path
import logging
from preprocess.data_loader import CsvLoader

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # loader = CsvLoader('/home/yichaoli8/fed_csis23')
    # write_dir = '/home/yichaoli8/fed_csis23/shards/train/'
    # for i, lower in enumerate(range(0, 16000, 2000)):
    #     # if i < 6:
    #     #     continue
    #     upper = lower + 2000
    #     df = loader.load_row_range('train', slice(lower, upper))
    #     fname = 'shard%i.parquet' % i
    #     write_path = os.path.join(write_dir, fname)
    #     logging.info('writing parquet %s' % write_path)
    #     df.to_parquet(write_path, engine='pyarrow', compression='gzip', index=False)
    #
    # pool.close()
    # pool.join()

    # make local train set
    loader = CsvLoader('/home/yichaoli8/fed_csis23')
    write_path = '/home/yichaoli8/fed_csis23/shards/train_local/shard0.parquet'
    df = loader.load_sample_csv(1000, 522)
    df.to_parquet(write_path, engine='pyarrow', compression='gzip', index=False)
