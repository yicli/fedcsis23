import logging
from multiprocessing import Pool
from preprocess.data_loader import ParquetLoader, one_hot_cols
from preprocess.features import Factorise

if __name__ == '__main__':

    def worker_func(c):
        loader = ParquetLoader('/home/yichaoli8/fed_csis23/shards/train')
        # logging.info('processing feature %s, column %s' % (f, c))
        data = loader.load_columns([c])
        fact = Factorise(c)
        fact.fit(data)
        fact.write_encoder()

    # Run in parallel
    # pool = Pool(4)
    # pool.map(worker_func, one_hot_cols)
    # pool.close()
    # pool.join()

    # Run one
    loader = ParquetLoader('/home/yichaoli8/fed_csis23/shards/train')
    c = 'PROCESS_PATH'
    logging.info('processing column %s' % c)
    data = loader.load_columns([c])
    fact = Factorise(c)
    fact.fit(data)
    fact.write_encoder()
