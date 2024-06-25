#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import List

from alibabacloud_sls20201230.client import Client as Sls20201230Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_sls20201230 import models as sls_20201230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
import os
import pandas as pd
from datetime import datetime, timedelta

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ALIBABA_CLOUD_ACCESS_KEY_ID = os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]

HOURS_TO_RUN = int(os.getenv("HOURS_TO_RUN", "8"))
MINUTES_INTERVAL = int(os.getenv("MINUTES_INTERVAL", "240"))
MINUTES_INTERVAL_SMALL = int(os.getenv("MINUTES_INTERVAL_SMALL", "15"))
ROUNDS_TO_RUN = int(os.getenv("ROUNDS_TO_RUN", "19"))

RUN_CONTINUOUSLY = "true" == os.getenv("RUN_CONTINUOUSLY", "false")

MYSQL_HOST = os.getenv("MYSQL_HOST", "0.0.0.0")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER_NAME = os.getenv("MYSQL_USER_NAME", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "peng_mbp13")

config = open_api_models.Config(
    access_key_id=ALIBABA_CLOUD_ACCESS_KEY_ID,
    access_key_secret=ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    read_timeout=120 * 1000,
    connect_timeout=10 * 1000,
    no_proxy="cn-hangzhou.log.aliyuncs.com",
)
# Endpoint 请参考 https://api.aliyun.com/product/Sls
config.endpoint = f"cn-hangzhou.log.aliyuncs.com"
sls_client = Sls20201230Client(config)
sls_client._read_timeout = 120 * 1000


# In[2]:


import pymysql
import threading


def write_data_with_sql(sql) -> int:
    # Create a connection to the MySQL server
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER_NAME,
        password=MYSQL_PASSWORD,
        database="grafana",
    )
    rows_inserted = 0
    cursor = conn.cursor()
    try:
        sql = sql.replace("'null'", "null")
        rows_inserted = cursor.execute(sql)
    except Exception as e:
        logging.error(f"sql:{sql}\nError inserting data: {e}")
        conn.rollback()
        raise e
    else:
        conn.commit()

    cursor.close()
    conn.close()
    return rows_inserted


WRITE_MYSQL_ROWS_BUFF = 500

# Dictionary to store locks for each resource
resource_locks = {}
lock_dict_lock = threading.Lock()

def get_lock_for_resource(resource_name):
    with lock_dict_lock:  # Ensure thread-safe access to the dictionary
        if resource_name not in resource_locks:
            resource_locks[resource_name] = threading.Lock()
        return resource_locks[resource_name]

def write_df_to_mysql(
    df: pd.DataFrame,
    table_name="product_views",
    columns=[
        "minute_of_day",
        "day",
        "time_of_day",
        "pdid",
        "sku",
        "name",
        "views",
        "avg_sale_price",
        "max_sale_price",
        "min_sale_price",
        "viewed_users",
    ],
    colomns_to_update=[
        "views",
        "avg_sale_price",
        "max_sale_price",
        "min_sale_price",
        "viewed_users",
    ],
) -> int:
    if df is None or columns is None or len(columns) <= 0:
        logging.warning("这是一个空数据集...无须插入")
        return 0
    
    # Get the lock for the specific resource
    resource_lock = get_lock_for_resource(table_name)
    
    # Use the lock in a context manager
    with resource_lock:
        df = df[columns]
        update_colomns = [f"{col}=values({col})" for col in colomns_to_update]
        update_colomns = ",".join(update_colomns)
        update_colomns = f"{update_colomns},update_time=now()"
        sql_template = f"""INSERT INTO {table_name}({','.join(columns)}) \nvalues __VALUES__ ON DUPLICATE KEY UPDATE 
        {update_colomns};"""
        values_list = []
        inserted_rows = 0
        for index, row in df.iterrows():
            value_of_row = []
            for col in columns:
                col_value_without_quote = f"{row[col]}".replace("'", "\\'")
                value_of_row.append(f"'{col_value_without_quote}'")
            values_list.append(f"""({','.join(value_of_row)})""")
            if len(values_list) >= WRITE_MYSQL_ROWS_BUFF:
                logging.info(f"about to write batch of data: {len(values_list)}")
                sql = sql_template.replace("__VALUES__", ",\n".join(values_list))
                inserted_rows = inserted_rows + write_data_with_sql(sql=sql)
                logging.info(f"write_data_with_sql() total afected so far:{inserted_rows}")
                values_list = []
        if len(values_list) > 0:
            logging.info(f"about to write last batch of data:{len(values_list)}")
            sql = sql_template.replace("__VALUES__", ",\n".join(values_list))
            if len(values_list) <= 10:
                logging.info(f"SQL监控:{sql}")
            inserted_rows = inserted_rows + write_data_with_sql(sql=sql)
            logging.info(f"write_data_with_sql() total afected so far:{inserted_rows}")
            values_list = []

        logging.info(f"写入了:{inserted_rows}条数据到MySQL")
        return inserted_rows


# In[3]:


# 获取SLS日志的查询结果
import time

headers = {
    "accept": "application/json",
    "user-agent": "AlibabaCloud API Workbench",
    "x-log-apiversion": "0.6.0",
    "x-log-bodyrawsize": "0",
    "x-log-signaturemethod": "hmac-sha256",
    "content-type": "application/json",
}


def get_sls_data_by_query(
    from_time: datetime,
    to_time: datetime,
    query="",
    project="xianmu-front-end-log",
    logstore="xm-mall",
    retry_time=1,
) -> pd.DataFrame:
    if retry_time < 0:
        logging.error(f"超过了最多重试次数")
        return None
    logging.info(
        f"即将获取数据: =====>from_time:{from_time}, to_time:{to_time}, logstore:{logstore}, query:{query[0:150]}",
    )

    from_ = int(from_time.timestamp())
    to_ = int(to_time.timestamp())

    get_logs_v2headers = sls_20201230_models.GetLogsV2Headers(
        common_headers=headers, accept_encoding="lz4"
    )
    get_logs_v2request = sls_20201230_models.GetLogsV2Request(
        from_=from_,
        to=to_,
        query=query,
    )
    runtime = util_models.RuntimeOptions(
        connect_timeout=10 * 1000,
        read_timeout=120 * 1000,
        no_proxy="cn-hangzhou.log.aliyuncs.com",
        max_attempts=2,
    )

    product_view_data = []
    try:
        response = sls_client.get_logs_v2with_options(
            project=project,
            logstore=logstore,
            request=get_logs_v2request,
            headers=get_logs_v2headers,
            runtime=runtime,
        )
        product_view_data = response.body.data
        logging.info(f">=====数据条数:{len(product_view_data)}")
        return pd.DataFrame(product_view_data)
    except Exception as error:
        logging.error(
            f"查询SLS失败了,重试:{retry_time},project:{project},logstore:{logstore}, 错误:{error}"
        )
        # 5 秒后重试一次
        time.sleep(5)
        return get_sls_data_by_query(
            from_time=from_time,
            to_time=to_time,
            query=query,
            project=project,
            logstore=logstore,
            retry_time=retry_time - 1,
        )


# In[4]:


# 获取SLS日志的查询结果

headers = {
    "accept": "application/json",
    "user-agent": "AlibabaCloud API Workbench",
    "x-log-apiversion": "0.6.0",
    "x-log-bodyrawsize": "0",
    "x-log-signaturemethod": "hmac-sha256",
    "content-type": "application/json",
}


def get_product_view_data_and_save(from_time: datetime, to_time: datetime):
    product_view_data_df = None
    query = """
    type:view|select * from (select *,row_number() over( partition by minute_of_day order by views desc ) as rnk 
    from (select minute_of_day,split(minute_of_day,' ')[1] as day,split(minute_of_day,' ')[2] as time_of_day,
        pdid,sku,name, count(0)views,round(avg(salePrice),2)avg_sale_price,max(salePrice) max_sale_price,
        min(salePrice) min_sale_price, count(distinct uid) as viewed_users 
        from (select uid,date_format(__time__,'%Y-%m-%d %H:%i:00') minute_of_day,pageName, 
            replace(regexp_extract(product_viewed, 'name:(?<name>[^,]+)'), 'name:','') as name, 
            replace(regexp_extract(product_viewed, 'sku:(?<sku>[^,]+)'), 'sku:','') as sku, 
            replace(regexp_extract(product_viewed, 'pdid:(?<pdid>[^,]+)'), 'pdid:','') as pdid, 
            cast(replace(regexp_extract(product_viewed, 'salePrice:(?<saleprice>[^,]+)'), 'salePrice:','') as double) as salePrice, 
            cast(replace(regexp_extract(product_viewed, 'stock:(?<stock>[^,]+)'), 'stock:','') as bigint) as stock 
            from log, unnest(split(bid,';')) as t(product_viewed) having sku is not null limit 1000000) 
            group by minute_of_day,pdid,sku,name)) 
        where rnk<=100 order by minute_of_day
    """
    # 商品的曝光数据只取top 1000的商品

    product_view_data_df = get_sls_data_by_query(
        from_time=from_time,
        to_time=to_time,
        query=query,
    )
    if product_view_data_df is None:
        logging.error(f"没有获取到商品的曝光数据:{from_time}~{to_time}")
        return
    # logging.info(f"过滤后的长度：{len(product_view_data_df)}")
    write_df_to_mysql(product_view_data_df)


def get_product_views_summary_and_save(from_time: datetime, to_time: datetime):
    df = None
    query = """
    type:view|select minute_of_day,case when stock>0 then 'normal' else 'out-of-stock' end as is_out_of_stock,
        count(0)views,
        count(distinct uid) as viewed_users,
        count(distinct sku) as viewed_skus
    from (select uid,date_format(__time__,'%Y-%m-%d %H:%i:00') minute_of_day,
            replace(regexp_extract(product_viewed, 'sku:(?<sku>[^,]+)'), 'sku:','') as sku,
            cast(replace(regexp_extract(product_viewed, 'stock:(?<stock>[^,]+)'), 'stock:','') as bigint) as stock
        from log, unnest(split(bid,';')) as t(product_viewed) having sku is not null limit 1000000) 
    group by minute_of_day,is_out_of_stock order by minute_of_day,is_out_of_stock limit 50000"""

    # 商品曝光数据，分钟级别汇总

    df = get_sls_data_by_query(
        from_time=from_time,
        to_time=to_time,
        query=query,
    )

    if df is None or len(df)<=0:
        logging.error(f"未获取到商品曝光数据:{from_time}, {to_time}")
        return

    write_df_to_mysql(
        df=df,
        table_name="product_views_summary",
        columns=[
            "minute_of_day",
            "is_out_of_stock",
            "views",
            "viewed_users",
            "viewed_skus",
        ],
        colomns_to_update=[
            "views",
            "viewed_users",
            "viewed_skus",
        ],
    )


# In[5]:


mall_backend_query = """
(inboundFlag:GET or inboundFlag:POST) and "xm-phone" not null | 
select second_of_hour,round(avg(uv)) avg_uv_30s,max(uv) max_uv_30s,round(avg(request_cnt)) avg_qps_30s,
    max(request_cnt) max_qps_30s,sum(request_cnt) request_cnt 
from (select count(distinct "xm-uid") as uv,count(distinct concat("xm-uid","xm-rqid")) as request_cnt,
    date_format(time, '%Y-%m-%d %H:%i:00') as second_of_hour,date_format(time, '%H:%i:%S') as second_of_hour_real 
from log group by second_of_hour,second_of_hour_real limit 360000) 
group by second_of_hour order by second_of_hour"""


def get_mall_qps_data_and_save(from_time: datetime, to_time: datetime):
    mall_qps_df = None
    mall_qps_df = get_sls_data_by_query(
        from_time=from_time,
        to_time=to_time,
        query=mall_backend_query,
        project="k8s-log-c7d28cba17d0a416ca4f52459592b8d38",
        logstore="prod-mall-stdout-log",
    )

    if mall_qps_df is None or len(mall_qps_df)<=0:
        logging.error(f"未获取到QPS数据:{from_time}, {to_time}")
        return

    write_df_to_mysql(
        mall_qps_df,
        table_name="mall_http_api_qps",
        columns=[
            "second_of_hour",
            "avg_uv_30s",
            "max_uv_30s",
            "avg_qps_30s",
            "max_qps_30s",
            "request_cnt",
        ],
        colomns_to_update=[
            "avg_uv_30s",
            "max_uv_30s",
            "avg_qps_30s",
            "max_qps_30s",
            "request_cnt",
        ],
    )


# In[6]:


sku_orders_query = """
ap:https\://h5.summerfarm.net/order/ |  select minute_of_day,json_extract_scalar(orderItems,'$.sku') sku,
    json_extract_scalar(orderItems,'$.pdName') pd_name ,json_extract_scalar(orderItems,'$.itemCategory') as category,
    count(distinct uid) ordered_users, sum(cast(json_extract_scalar(orderItems,'$.amount') as bigint)) as total_quantity ,
    round(sum(cast(json_extract_scalar(orderItems,'$.actualTotalPrice') as double)),2) as total_gmv 
from (
    select orderItems,date_format(__time__,'%Y-%m-%d %H:%i:00') as minute_of_day,uid 
        from log, unnest(cast(json_extract(ai, '$.rt.data.orderItems') as array(json))) as t(orderItems)  
        having orderItems is not null limit 1000000) 
group by minute_of_day,sku,pd_name,category order by minute_of_day,total_gmv desc limit 300000"""


def get_product_orders_and_save(from_time: datetime, to_time: datetime):
    orders_df = None
    orders_df = get_sls_data_by_query(
        from_time=from_time,
        to_time=to_time,
        query=sku_orders_query,
        project="xianmu-front-end-log",
        logstore="xm-mall",
    )

    if orders_df is None or len(orders_df)<=0:
        logging.error(f"未获取到前端上报到订单数据:{from_time}, {to_time}")
        return

    write_df_to_mysql(
        orders_df,
        table_name="product_orders",
        columns=[
            "minute_of_day",
            "sku",
            "pd_name",
            "category",
            "ordered_users",
            "total_quantity",
            "total_gmv",
        ],
        colomns_to_update=["ordered_users", "total_quantity", "total_gmv"],
    )


# In[7]:


from datetime import datetime, timedelta
from concurrent import futures
import concurrent
import time


def process_timerange(start_time_of_run, end_time_of_run) -> bool:
    try:
        logging.info("获取分钟级别汇总的商品曝光数")
        get_product_views_summary_and_save(start_time_of_run, end_time_of_run)

        logging.info("获取商品曝光数据:")
        get_product_view_data_and_save(start_time_of_run, end_time_of_run)

        logging.info("获取后端QPS数据")
        get_mall_qps_data_and_save(start_time_of_run, end_time_of_run)

        logging.info("获取商品下单数据：前端SLS日志")
        get_product_orders_and_save(start_time_of_run, end_time_of_run)
        return True
    except Exception as e:
        logging.error(
            f"发生了异常, start_time_of_run:{start_time_of_run}, end_time_of_run:{end_time_of_run}, error:{e}"
        )
        return False


running_counter = 1
INIT_LAST_7DAYS_DATA = os.getenv("INIT_LAST_7DAYS_DATA", "false")
INIT_LAST_7DAYS_DATA = f"{INIT_LAST_7DAYS_DATA}" == "true"
PYTHON_RUN_INTERVAL = int(os.getenv("PYTHON_RUN_INTERVAL", "45"))
INIT_LAST_N_DAYS = int(os.getenv("INIT_LAST_N_DAYS", "3"))
while running_counter > 0:
    # 对时间的秒数取整，比如 2024-01-01 00:01:00
    now = datetime.now().strftime("%Y-%m-%d %H:%M:00")
    now = datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
    end_time_of_run = now
    if INIT_LAST_7DAYS_DATA:
        MINUTES_INTERVAL_SMALL = MINUTES_INTERVAL
        ROUNDS_TO_RUN = int(24 * 60 / MINUTES_INTERVAL * INIT_LAST_N_DAYS + 1)

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_list=[]
        for interval in range(1, ROUNDS_TO_RUN):
            start_time_of_run = end_time_of_run - timedelta(minutes=MINUTES_INTERVAL_SMALL)
            future_list.append(executor.submit(process_timerange, start_time_of_run, end_time_of_run))
            end_time_of_run = start_time_of_run
        
        concurrent.futures.wait(future_list)
        
    if "false" == f"{RUN_CONTINUOUSLY}".lower():
        logging.info(f"系统没有设置持续性的跑,RUN_CONTINUOUSLY:{RUN_CONTINUOUSLY}")
        break
    else:
        logging.info(
            f"是否持续性的跑:{RUN_CONTINUOUSLY},running_counter:{running_counter}, PYTHON_RUN_INTERVAL:{PYTHON_RUN_INTERVAL}s"
        )
        running_counter = running_counter + 1
        time.sleep(PYTHON_RUN_INTERVAL)
        INIT_LAST_7DAYS_DATA = False
        MINUTES_INTERVAL_SMALL = 30
        ROUNDS_TO_RUN = 2


# In[ ]:





# In[ ]:




