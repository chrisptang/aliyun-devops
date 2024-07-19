import time
from datetime import datetime, timedelta
import pandas as pd
import os
from odps import ODPS, DataFrame
from odps.accounts import StsAccount
import traceback
import logging

app_log_dir = os.environ.get("APP_LOG_DIR", "./")

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{app_log_dir}/app.log"),  # Logs to a file named 'app.log'
        logging.StreamHandler(),  # Logs to the console
    ],
)

import json

logging.info(json.dumps(dict(os.environ), indent=2))

ALIBABA_CLOUD_ACCESS_KEY_ID = os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]
THREAD_CNT = int(os.environ.get("THREAD_CNT", 20))
USE_PROXY = bool(os.environ.get("USE_PROXY", True))

DEFAULT_MAX_RETRY_NUM = 5

logging.info(f"Thread count: {THREAD_CNT}")

odps = ODPS(
    ALIBABA_CLOUD_ACCESS_KEY_ID,
    ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    project="summerfarm_ds_dev",
    endpoint="http://service.cn-hangzhou.maxcompute.aliyun.com/api",
)


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


from datetime import datetime

time_of_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_of_now = datetime.now().strftime("%Y-%m-%d")


hints = {"odps.sql.hive.compatible": True, "odps.sql.type.system.odps2": True}


def get_odps_sql_result_as_df(sql) -> pd.DataFrame:
    instance = odps.execute_sql(sql, hints=hints)
    instance.wait_for_success()
    pd_df = None
    with instance.open_reader(tunnel=True) as reader:
        # type of pd_df is pandas DataFrame
        pd_df = reader.to_pandas()

    if pd_df is not None:
        logging.info(f"sql:\n{sql}\ncolumns:{pd_df.columns}")
        return pd_df
    return None


def add_new_column_to_table(table_name, column_name):
    if not "summerfarm_ds." in table_name:
        table_name = f"summerfarm_ds.{table_name}"
    sql = f"ALTER TABLE {table_name} ADD COLUMNS ({column_name} STRING);"
    instance = odps.execute_sql(sql)
    instance.wait_for_success()
    logging.info(f"添加新字段成功:{table_name}, {column_name}")


def ensure_all_df_columns_in_odps_table(df, table_name):
    if not "summerfarm_ds." in table_name:
        table_name = f"summerfarm_ds.{table_name}"
    if not odps.exist_table(table_name):
        logging.info(f"表不存在:{table_name}")
        return True
    table = odps.get_table(table_name)
    column_names = set([column.name for column in table.table_schema])
    column_names_out = ",".join(column_names)
    logging.info(f"DaraFrame字段合集:{column_names_out}")
    df_columns = df.columns.tolist()
    for df_col in df_columns:
        df_col = df_col.lower()
        if not df_col in column_names:
            logging.info(f"新字段:{df_col}, ODPS全部的字段:{column_names}")
            add_new_column_to_table(table_name, df_col)
    return True


def write_pandas_df_into_odps(df, table_name, partition_spec) -> bool:
    if df is None or len(df) <= 0:
        logging.info(f"数据DF为空, table:{table_name}")
        return False
    time_of_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["spider_fetch_time"] = time_of_now
    ensure_all_df_columns_in_odps_table(df, table_name)
    exception = None
    for attemp in range(5):
        try:
            odps_df = DataFrame(df)
            odps_df.persist(
                table_name,
                partition=partition_spec,
                drop_partition=False,
                create_partition=True,
                overwrite=False,
                lifecycle=365,
            )
            logging.info(
                f"成功写入odps:{table_name}, partition_spec:{partition_spec}, attemp:{attemp}"
            )
            # 返回true
            return True
        except Exception as e:
            exception = e
            logging.info(f"写入ODPS不成功:{table_name}{e}")
            traceback.print_exc()
            time.sleep(10)
    # 默认返回false
    if exception is not None:
        raise exception
    return False
