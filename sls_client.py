import os
import traceback

from alibabacloud_sls20201230.client import Client as Sls20201230Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_sls20201230 import models as sls_20201230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
import os
import pandas as pd
from datetime import datetime, timedelta
import time

ALIBABA_CLOUD_ACCESS_KEY_ID = os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"]
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"]

HOURS_TO_RUN = int(os.getenv("HOURS_TO_RUN", "8"))
HOURS_INTERVAL = int(os.getenv("HOURS_INTERVAL", "4"))
HOURS_INTERVAL_SMALL = int(os.getenv("HOURS_INTERVAL_SMALL", "4"))
ROUNDS_TO_RUN = int(os.getenv("ROUNDS_TO_RUN", "19"))

MYSQL_HOST = os.getenv("MYSQL_HOST", "0.0.0.0")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER_NAME = os.getenv("MYSQL_USER_NAME", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "peng_mbp13")

config = open_api_models.Config(
    access_key_id=ALIBABA_CLOUD_ACCESS_KEY_ID,
    access_key_secret=ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    read_timeout=120 * 1000,
    connect_timeout=10 * 1000,
)
# Endpoint 请参考 https://api.aliyun.com/product/Sls
config.endpoint = f"cn-hangzhou.log.aliyuncs.com"
sls_client = Sls20201230Client(config)

# 获取SLS日志的查询结果

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
    project="k8s-log-c7d28cba17d0a416ca4f52459592b8d38",
    logstore="prod-mall-stdout-log",
) -> pd.DataFrame:
    print(
        "即将获取数据: =====>",
        from_time,
        to_time,
        f"{logstore}: {query[0:100]}",
    )

    from_time = int(from_time.timestamp())
    to_time = int(to_time.timestamp())

    get_logs_v2headers = sls_20201230_models.GetLogsV2Headers(
        common_headers=headers, accept_encoding="lz4"
    )
    get_logs_v2request = sls_20201230_models.GetLogsV2Request(
        from_=from_time,
        to=to_time,
        query=query,
    )
    runtime = util_models.RuntimeOptions()

    sls_query_data = []
    try:
        response = sls_client.get_logs_v2with_options(
            project=project,
            logstore=logstore,
            request=get_logs_v2request,
            headers=get_logs_v2headers,
            runtime=runtime,
        )
        sls_query_data = response.body.data
        print(f">=====数条数:{len(sls_query_data)}")
    except Exception as error:
        print(f"sql:{query}, 查询SLS失败了:{error}:{traceback.format_exc()}")
    return pd.DataFrame(sls_query_data)

def get_sls_raw_data_by_query(
    from_time: datetime,
    to_time: datetime,
    query: str = "",
    project: str = "xianmu-front-end-log",
    logstore: str = "xm-mall",
    retry_time: int = 1,
    line: int = 1000,
    offset: int = 0,
) -> pd.DataFrame:
    if retry_time < 0:
        print(f"超过了最多重试次数")
        return None
    if offset % 10000 == 0:
        print(
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
        line=line,
        offset=offset,
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
        if offset % 10000 == 0:
            print(f">=====数据条数:{len(product_view_data)}")
        return pd.DataFrame(product_view_data)
    except Exception as error:
        print(
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
