from ast import literal_eval
import pandas as pd
import pyarrow.parquet as pa


def get_columns_by_lv0(schema_names, level_0):
    """
    Return columns from schema that matches level_0
    :param schema_names: schema names of a parquet file
    :param level_0: the level 0 names to load
    :return: list of full schema names with matching level 0
    """
    schema_df = pd.DataFrame([literal_eval(n) for n in schema_names])
    mask = schema_df[0].isin(level_0)
    cols_to_load = pd.Series(schema_names)[mask]
    return list(cols_to_load)


def get_feature_from_parquet(parquet_dir):
    ds = pa.ParquetDataset(parquet_dir)
    schema_df = pd.DataFrame([literal_eval(n) for n in ds.schema.names])
    return schema_df[0].unique()


if __name__ == '__main__':
    get_feature_from_parquet('../features/train_local_scaled')