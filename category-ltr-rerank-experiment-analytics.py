#!/usr/bin/env python
# coding: utf-8



import argparse
import os
import glob as glob_module

# 解析命令行参数
parser = argparse.ArgumentParser(description='分类页LTR重排序AB实验分析')
parser.add_argument('--experimental-id', type=str, default='ltr_rerank', help='实验ID (默认: ltr_rerank)')
parser.add_argument('--start-date', type=str, default='2026-01-07', help='开始日期，格式YYYY-MM-DD (默认: 2026-01-07)')
parser.add_argument('--reload-all', action='store_true', help='移除所有本地sqlite文件，重新获取数据')
parser.add_argument('--upload-odps', action='store_true', help='将数据上传到ODPS表')

args = parser.parse_args()

EXPERIMENTAL_ID = args.experimental_id
START_DATE = args.start_date

# 如果指定了--reload-all，删除本地sqlite文件
if args.reload_all:
    sqlite_files = glob_module.glob('./data/category_ltr_ab_*.db')
    for f in sqlite_files:
        print(f"删除本地缓存文件: {f}")
        os.remove(f)
    print(f"已删除 {len(sqlite_files)} 个本地sqlite文件")




from odps_client import get_odps_sql_result_as_df
from datetime import datetime, timedelta




from sls_client import get_sls_data_by_query
from datetime import datetime, timedelta
import pandas as pd

# 设置pandas显示选项以展示更多内容
pd.set_option("display.max_rows", 100)  # 显示最多100行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 1000)  # 设置显示宽度
pd.set_option("display.max_colwidth", 100)  # 设置列最大宽度

import sqlite3

# ============================================================================
# 全局matplotlib字体配置 - 支持中文显示（Mac系统优化）
# ============================================================================
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，确保字体配置生效

if sys.platform == 'darwin':
    # Mac系统：使用系统内置字体（根据实际可用字体调整）
    # 可用字体: Heiti TC, PingFang HK, STHeiti, Songti SC, Kaiti SC
    matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'STHeiti', 'Songti SC', 'Kaiti SC', 'Arial Unicode MS']
else:
    # 其他系统（Windows/Linux）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']

# 统一配置
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'

import matplotlib.pyplot as plt


def get_user_variant_of_date_from_sls(
    day: datetime, check_if_local_exist: bool = True
) -> pd.DataFrame:
    """
    从SLS(Simple Log Service)获取指定日期的用户变体数据。

    Args:
        day (datetime): 要获取数据的日期。
        check_if_local_exist (bool): 是否检查本地数据库中是否已存在数据，默认为True。

    Returns:
        pd.DataFrame: 包含用户变体数据的DataFrame。
    """
    # 构建数据库文件名和表名
    db_file_name = f"./data/category_ltr_ab_user_variant.db"
    table_name = f"recommend_ab_user_variant_{day.strftime('%Y%m%d')}"
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file_name)

    # 如果设置为检查本地数据
    if check_if_local_exist:
        try:
            # 尝试从数据库中读取数据
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            # 关闭数据库连接
            conn.close()
            # 返回读取的数据
            return df
        except pd.io.sql.DatabaseError:
            # 如果表不存在，则忽略错误
            pass

    # 构建SLS查询语句
    query = f"""
type:a and ap:/product
| SELECT 
    regexp_replace(ap, '\d+','{{digit}}') as api,
    case when pageName in ('/goods','/goods/category') then '/goods' else 'null' end as page_name,
    regexp_extract(experiment_item, '"experimentId":"([^"]+)"', 1) as experiment_id,
    type,
    uid,
    date_format(__time__, '%Y%m%d') as ds,
    count(1) as search_times,
    array_join(array_sort(array_agg(distinct regexp_extract(experiment_item, '"variantId":"([^"]+)"', 1))),',') as variant_list
FROM log, 
UNNEST(regexp_extract_all(json_extract_scalar(ai, '$.qh.xm-ab-exp'), '\{{[^}}]+\}}')) as t(experiment_item)
WHERE experiment_item LIKE '%"experimentId":"{EXPERIMENTAL_ID}"%'
GROUP BY 1,2,3,4,5,6
HAVING page_name = '/goods'
LIMIT 1000000
"""
    # 设置查询的起始时间和结束时间
    print(query)
    from_time = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_time = day.replace(hour=23, minute=59, second=59, microsecond=999999)
    # 从SLS获取数据
    _df = get_sls_data_by_query(
        query=query,
        project="xianmu-front-end-log",  # 指定SLS项目
        logstore="xm-mall",  # 指定SLS日志库
        from_time=from_time,  # 指定查询起始时间
        to_time=to_time,  # 指定查询结束时间
    )

    # 将search_times列中的缺失值填充为1，并转换为整数类型
    _df["search_times"] = _df["search_times"].fillna(1).astype(int)
    # 将variant_list列中的缺失值填充为"none"
    _df["variant_list"] = _df["variant_list"].fillna("none")

    # 如果DataFrame不为空
    if not _df.empty:
        # 删除不需要的列
        _df.drop(columns=["__source__", "__time__"], inplace=True)
        # 将数据写入SQLite数据库，如果表已存在则替换
        _df.to_sql(table_name, conn, if_exists="replace", index=False)
    # 关闭数据库连接
    conn.close()
    # 返回数据
    return _df


# 创建一个空的DataFrame来存储所有日期的用户变体数据
all_user_variant_df = pd.DataFrame()
# 设置起始日期和结束日期
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.now()
# 从起始日期开始循环，直到结束日期
current_date = start_date
while current_date <= end_date:
    # 检查是否是今天
    is_today = current_date.strftime("%Y%m%d") == end_date.strftime("%Y%m%d")
    # 如果是今天,则跳过，因为今天的数据可能不完整
    if is_today:
        print(f"今天的数据还未完整，跳过:{current_date}")
        break
    # 获取当前日期的用户变体数据
    df = get_user_variant_of_date_from_sls(current_date, check_if_local_exist=True)
    # 将当前日期的数据添加到总的DataFrame中
    all_user_variant_df = pd.concat([all_user_variant_df, df], ignore_index=True)
    # 日期增加一天
    current_date += timedelta(days=1)

# 显示前10行数据
all_user_variant_df.head(10)




import pandasql
stats=pandasql.sqldf("""select ds,variant_list,count(distinct uid) unique_user 
                     from all_user_variant_df group by ds,variant_list order by ds desc,variant_list""")

print(stats)




# idx:4,name:徐州奶油草莓 净重3-3.2斤/一级/单果10g+/板装,pid:goods,sku:5442468008,salePrice:69.5,pdid:702,stock:10000,ext:cross;idx:5,name:徐州奶油草莓 280G*1盒/一级/4*6/ ,pid:goods,sku:5442468073,salePrice:16.5,pdid:702,stock:10000,ext:cross;idx:6,name:徐州奶油草莓 净重2.8-3斤/一级/单果10g+/ 10盒,pid:goods,sku:5442468518,salePrice:74.5,pdid:702,stock:10000,ext:cross

view_query = """
type:cl or type:view | select ds,
    '/goods' as "页面名称",uid,sku_viewed_or_clicked,cate_level1_id,cate_level2_id,
    count(distinct case when type='view' and position_id='goods' then (uid,sku_viewed_or_clicked,__time__,idx) end) as "SKU曝光次数",
    count(distinct case when type='cl' and position_id in ('goods','唤起购买') then (uid,sku_viewed_or_clicked,__time__,idx) end) as "SKU点击次数",
    count(distinct case when type='cl' and position_id in ('goods','唤起购买') and idx <= 3 then (uid,sku_viewed_or_clicked,__time__,idx) end) as "前4位SKU点击次数",
    count(distinct case when type='cl' and position_id='唤起购买' then (uid,sku_viewed_or_clicked,__time__,idx) end) as "唤起加购弹窗点击次数",
    count(distinct case when type='cl' and position_id='加购弹窗' and position_name='加入购物车' then (uid,sku_viewed_or_clicked,__time__,idx) end) as "加入购物车次数",
    count(distinct case when type='cl' and position_id='加购弹窗' and position_name='立即购买' then (uid,sku_viewed_or_clicked,__time__,idx) end) as "立即购买次数",
    round(avg(case when type='cl' then idx end),2) as "平均点击位置",
    max(case when type='cl' then idx end) as "最大点击位置",
    min(case when type='cl' then idx end) as "最小点击位置",
    round(avg(case when type='view' and position_id='goods' then idx end),2) as "平均曝光位置",
    max(case when type='view' and position_id='goods' then idx end) as "最大曝光位置",
    min(case when type='view' and position_id='goods' then idx end) as "最小曝光位置",
    count(0) as "总事件数" 
from (
    select pageName,
        type,
        case when COALESCE(regexp_extract(bid, 'pid:([^,]+)', 1),pid) in ('goods','唤起购买') and type='cl' 
            then COALESCE(regexp_extract(bid, 'pid:([^,]+)', 1),pid) 
            else COALESCE(regexp_extract(bid, 'name:([^,]+)', 1),name) 
        end as position_name,
        regexp_extract(url, '[?&]id=([^&]+)', 1) AS cate_level1_id,
    	regexp_extract(url, '[?&]cid=([^&]+)', 1) AS cate_level2_id,
        COALESCE(regexp_extract(bid, 'pid:([^,]+)', 1),pid) as position_id,
        uid,__time__,
        cast(COALESCE(regexp_extract(bid, 'idx:([0-9]+)', 1),idx) as bigint) as idx,
        COALESCE(regexp_extract(bid, 'sku:([0-9a-zA-Z]{1,20})', 1),sku) as sku_viewed_or_clicked,
        date_format(__time__,'%Y%m%d') ds
    from log 
    where pageName in ('/goods','/goods/category')
        and COALESCE(regexp_extract(bid, 'sku:([0-9a-zA-Z]{1,20})', 1),sku) is not null 
    limit 100000000
)
group by 1,2,3,4,5,6
having length(uid)>0
order by 7 desc
"""


def get_user_sku_view_of_date_from_sls(
    day: datetime, check_if_local_exist: bool = True
) -> pd.DataFrame:
    db_file_name = f"./data/category_ltr_ab_user_sku_view.db"
    table_name = f"user_sku_view_{day.strftime('%Y%m%d')}"
    conn = sqlite3.connect(db_file_name)

    if check_if_local_exist:
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except pd.io.sql.DatabaseError:
            pass

    from_time = day.replace(hour=0, minute=0, second=0, microsecond=0)
    to_time = day.replace(hour=23, minute=59, second=59, microsecond=999999)
    _df = get_sls_data_by_query(
        query=view_query,
        project="xianmu-front-end-log",
        logstore="xm-mall",
        from_time=from_time,
        to_time=to_time,
    )

    if not _df.empty:
        _df.drop(columns=["__source__", "__time__"], inplace=True)
        _df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return _df


all_user_sku_view_df = pd.DataFrame()
current_date = start_date
while current_date <= end_date:
    is_today = current_date.strftime("%Y%m%d") == end_date.strftime("%Y%m%d")
    if is_today:
        print(f"今天的数据还未完整，跳过:{current_date}")
        break
    df = get_user_sku_view_of_date_from_sls(current_date, check_if_local_exist=True)
    all_user_sku_view_df = pd.concat([all_user_sku_view_df, df], ignore_index=True)
    current_date += timedelta(days=1)




all_user_sku_view_df.head(20)




all_user_sku_view_df.groupby("ds").size().reset_index(name="count").sort_values(
    "ds", ascending=False
).head(10)

user_click_with_variant_statistics_df = all_user_sku_view_df.merge(
    all_user_variant_df[["uid", "ds", "variant_list"]],
    on=["uid", "ds"],
    how="left",
)

print("non variant count:", user_click_with_variant_statistics_df[
    user_click_with_variant_statistics_df["variant_list"].isna()
].shape[0])

print("total count:", len(user_click_with_variant_statistics_df))

print("variant count:", len(user_click_with_variant_statistics_df[
    user_click_with_variant_statistics_df["variant_list"].notna()
]))

variant_ratio = len(user_click_with_variant_statistics_df[
    user_click_with_variant_statistics_df["variant_list"].notna()
]) / len(user_click_with_variant_statistics_df)
print("variant ratio:", f"{variant_ratio:.4%}")

user_click_with_variant_statistics_df.head(5)

print("drop non variant users...")
user_click_with_variant_statistics_df = user_click_with_variant_statistics_df[
    user_click_with_variant_statistics_df["variant_list"].notna()
]


# ============================================================================
# 获取用户画像数据：通过ODPS临时表LEFT JOIN（一次性获取）
# ============================================================================

def get_user_profile_via_odps_join(uids: list) -> pd.DataFrame:
    """
    将uid列表上传到ODPS临时表，然后用LEFT JOIN一次性获取所有用户画像

    Args:
        uids: 用户uid列表

    Returns:
        pd.DataFrame: 包含cust_id, region, has_trd_amt_90d的DataFrame
    """
    from odps_client import write_pandas_df_into_odps

    if not uids:
        return pd.DataFrame(columns=['cust_id', 'region', 'has_trd_amt_90d'])

    # 1. 创建uid DataFrame并上传到ODPS临时表
    uid_df = pd.DataFrame({'cust_id': [str(uid) for uid in uids]})
    temp_table = "summerfarm_ds.temp_ab_experiment_uids"
    # 使用当天日期作为分区
    partition_spec = f"ds={datetime.now().strftime('%Y%m%d')}"

    print(f"正在上传 {len(uids)} 个用户ID到ODPS临时表...")
    write_pandas_df_into_odps(
        df=uid_df,
        table_name=temp_table,
        partition_spec=partition_spec,
        overwrite=True,
        lifecycle=1  # 1天后自动删除
    )
    print(f"✅ 用户ID已上传到临时表: {temp_table}")

    # 2. 执行LEFT JOIN SQL获取用户画像
    today_ds = datetime.now().strftime('%Y%m%d')
    sql = f"""
    SELECT
        t.cust_id,
        CASE
            WHEN p.register_province IN ('浙江', '浙江省', '上海', '上海市', '江苏', '江苏省') THEN '华东区'
            WHEN p.register_province IN ('广西', '广西壮族自治区', '广东', '广东省') THEN '华南区'
            WHEN p.register_province IN ('湖北', '湖北省', '湖南', '湖南省', '江西', '江西省') THEN '华中区'
            ELSE '其他区域'
        END AS region,
        CASE WHEN p.trd_amt_90d > 0 THEN '90天内有交易额' ELSE '无' END AS has_trd_amt_90d
    FROM {temp_table} t
    LEFT JOIN summerfarm_tech.dws_cust_profile_asset_df p
        ON t.cust_id = p.cust_id
        AND p.ds = max_pt('summerfarm_tech.dws_cust_profile_asset_df')
    WHERE t.ds = '{today_ds}'
    """

    print("正在从ODPS获取用户画像数据（LEFT JOIN）...")
    result_df = get_odps_sql_result_as_df(sql)
    print(f"✅ 获取到 {len(result_df)} 条用户画像记录")

    return result_df


# 获取实验用户的uid列表
experiment_uids = user_click_with_variant_statistics_df['uid'].unique().tolist()
print(f"\n实验用户数: {len(experiment_uids)}")

# 从ODPS获取用户画像（一次性LEFT JOIN）
print("\n" + "=" * 60)
print("获取用户画像数据（区域 & 90天交易属性）")
print("=" * 60)
user_profile_df = get_user_profile_via_odps_join(experiment_uids)

# 将用户画像数据合并到实验数据中
user_click_with_variant_statistics_df = user_click_with_variant_statistics_df.merge(
    user_profile_df[['cust_id', 'region', 'has_trd_amt_90d']],
    left_on='uid',
    right_on='cust_id',
    how='left'
)

# 填充缺失值
user_click_with_variant_statistics_df['region'] = user_click_with_variant_statistics_df['region'].fillna('其他区域')
user_click_with_variant_statistics_df['has_trd_amt_90d'] = user_click_with_variant_statistics_df['has_trd_amt_90d'].fillna('无')

# 显示区域分布
print("\n实验用户区域分布:")
print(user_click_with_variant_statistics_df.groupby('region')['uid'].nunique())

# 显示90天交易属性分布
print("\n实验用户90天交易属性分布:")
print(user_click_with_variant_statistics_df.groupby('has_trd_amt_90d')['uid'].nunique())


print(user_click_with_variant_statistics_df.columns)




user_click_with_variant_statistics_df.head(20)

# 数据类型转换：使用pd.to_numeric安全转换，无法转换的值变为NaN
# 必须在所有聚合操作之前完成类型转换
user_click_with_variant_statistics_df['SKU点击次数'] = pd.to_numeric(
    user_click_with_variant_statistics_df['SKU点击次数'],
    errors='coerce'
).fillna(0).astype(int)

user_click_with_variant_statistics_df['SKU曝光次数'] = pd.to_numeric(
    user_click_with_variant_statistics_df['SKU曝光次数'],
    errors='coerce'
).fillna(0).astype(int)

user_click_with_variant_statistics_df['平均点击位置'] = pd.to_numeric(
    user_click_with_variant_statistics_df['平均点击位置'],
    errors='coerce'
)  # 保留NaN，后续过滤时使用

# 创建用户维度的画像映射表（用于后续JOIN）
user_profile_mapping = (
    user_click_with_variant_statistics_df[['uid', 'region', 'has_trd_amt_90d']]
    .drop_duplicates(subset=['uid'])
)

# 聚合1: SKU曝光次数
user_sku_view_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({"SKU曝光次数": "sum"})
    .reset_index()
)
# 关联用户画像
user_sku_view_serial = user_sku_view_serial.merge(user_profile_mapping, on='uid', how='left')

user_sku_view_serial.head(5)

# 聚合2: 平均点击位置 (过滤掉无效数据)
user_avg_position_serial = (
    user_click_with_variant_statistics_df[
        (user_click_with_variant_statistics_df['SKU点击次数'] > 0) &
        (user_click_with_variant_statistics_df['平均点击位置'].notna())
    ]
    .groupby(["variant_list", "uid"])
    .agg({"平均点击位置": "mean"})
    .reset_index()
)
# 关联用户画像
user_avg_position_serial = user_avg_position_serial.merge(user_profile_mapping, on='uid', how='left')

user_avg_position_serial.head(5)

# 聚合3: SKU点击次数
user_sku_click_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({"SKU点击次数": "sum"})
    .reset_index()
)
# 关联用户画像
user_sku_click_serial = user_sku_click_serial.merge(user_profile_mapping, on='uid', how='left')

user_sku_click_serial.head(5)




# 聚合4: 用户级别CTR计算
# 计算方式：先按用户聚合总点击次数和总曝光次数，再计算 CTR = sum(点击) / sum(曝光)
# 这样可以避免单个SKU级别的异常数据（如只有点击没有曝光）对整体CTR的影响
user_sku_ctr_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({
        "SKU点击次数": "sum",
        "SKU曝光次数": "sum"
    })
    .reset_index()
)

# 过滤掉曝光次数为0的用户（避免除以零）
user_sku_ctr_serial = user_sku_ctr_serial[user_sku_ctr_serial['SKU曝光次数'] > 0].copy()

# 计算用户级别CTR: 用户总点击次数 / 用户总曝光次数
user_sku_ctr_serial['CTR'] = user_sku_ctr_serial['SKU点击次数'] / user_sku_ctr_serial['SKU曝光次数']

# 关联用户画像
user_sku_ctr_serial = user_sku_ctr_serial.merge(user_profile_mapping, on='uid', how='left')

user_sku_ctr_serial.head(5)


# ============================================================================
# 聚合5-8: 按类目维度的用户级别聚合（用于类目分组分析）
# ============================================================================

# 创建类目维度的映射表（用于后续JOIN）
user_cate_mapping = (
    user_click_with_variant_statistics_df[['uid', 'cate_level1_id', 'cate_level2_id']]
    .drop_duplicates()
)

# 聚合5: 按一级类目的SKU曝光次数
user_sku_view_by_cate1 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level1_id"])
    .agg({"SKU曝光次数": "sum"})
    .reset_index()
)

# 聚合6: 按一级类目的SKU点击次数
user_sku_click_by_cate1 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level1_id"])
    .agg({"SKU点击次数": "sum"})
    .reset_index()
)

# 聚合7: 按一级类目的平均点击位置
user_avg_position_by_cate1 = (
    user_click_with_variant_statistics_df[
        (user_click_with_variant_statistics_df['SKU点击次数'] > 0) &
        (user_click_with_variant_statistics_df['平均点击位置'].notna())
    ]
    .groupby(["variant_list", "uid", "cate_level1_id"])
    .agg({"平均点击位置": "mean"})
    .reset_index()
)

# 聚合8: 按一级类目的CTR
user_ctr_by_cate1 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level1_id"])
    .agg({
        "SKU点击次数": "sum",
        "SKU曝光次数": "sum"
    })
    .reset_index()
)
user_ctr_by_cate1 = user_ctr_by_cate1[user_ctr_by_cate1['SKU曝光次数'] > 0].copy()
user_ctr_by_cate1['CTR'] = user_ctr_by_cate1['SKU点击次数'] / user_ctr_by_cate1['SKU曝光次数']

# 聚合9-12: 按二级类目的聚合
user_sku_view_by_cate2 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level2_id"])
    .agg({"SKU曝光次数": "sum"})
    .reset_index()
)

user_sku_click_by_cate2 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level2_id"])
    .agg({"SKU点击次数": "sum"})
    .reset_index()
)

user_avg_position_by_cate2 = (
    user_click_with_variant_statistics_df[
        (user_click_with_variant_statistics_df['SKU点击次数'] > 0) &
        (user_click_with_variant_statistics_df['平均点击位置'].notna())
    ]
    .groupby(["variant_list", "uid", "cate_level2_id"])
    .agg({"平均点击位置": "mean"})
    .reset_index()
)

user_ctr_by_cate2 = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid", "cate_level2_id"])
    .agg({
        "SKU点击次数": "sum",
        "SKU曝光次数": "sum"
    })
    .reset_index()
)
user_ctr_by_cate2 = user_ctr_by_cate2[user_ctr_by_cate2['SKU曝光次数'] > 0].copy()
user_ctr_by_cate2['CTR'] = user_ctr_by_cate2['SKU点击次数'] / user_ctr_by_cate2['SKU曝光次数']

print(f"\n📊 一级类目数量: {user_click_with_variant_statistics_df['cate_level1_id'].nunique()}")
print(f"📊 二级类目数量: {user_click_with_variant_statistics_df['cate_level2_id'].nunique()}")


# ## AB实验统计分析模块



import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ABTestResult:
    """AB测试结果数据类"""
    metric_name: str
    variant: str
    control: str
    sample_size_variant: int
    sample_size_control: int
    mean_variant: float
    mean_control: float
    lift: float  # 提升度 (variant - control) / control
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float  # Cohen's d
    power: float  # 统计功效


def calculate_descriptive_stats(data: pd.Series) -> Dict:
    """
    计算描述性统计指标

    Args:
        data: 数据序列

    Returns:
        包含各统计指标的字典
    """
    return {
        '样本数': len(data),
        '均值': data.mean(),
        '标准差': data.std(),
        '方差': data.var(),
        '最小值': data.min(),
        '最大值': data.max(),
        'P50': data.quantile(0.50),
        'P90': data.quantile(0.90),
        'P95': data.quantile(0.95),
        'P99': data.quantile(0.99),
    }


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    计算Cohen's d效应量

    Args:
        group1: 实验组数据
        group2: 对照组数据

    Returns:
        Cohen's d值
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def calculate_statistical_power(
    effect_size: float,
    n1: int,
    n2: int,
    alpha: float = 0.05
) -> float:
    """
    计算统计功效 (基于正态分布近似)

    Args:
        effect_size: Cohen's d效应量
        n1: 实验组样本量
        n2: 对照组样本量
        alpha: 显著性水平

    Returns:
        统计功效值
    """
    # 有效样本量
    n_eff = (n1 * n2) / (n1 + n2)

    # 非中心参数
    ncp = effect_size * np.sqrt(n_eff)

    # 临界值
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # 功效计算
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    return power


def perform_ab_test(
    variant_data: pd.Series,
    control_data: pd.Series,
    metric_name: str,
    variant_name: str = "实验组",
    control_name: str = "V2",
    alpha: float = 0.05
) -> ABTestResult:
    """
    执行AB测试统计分析

    Args:
        variant_data: 实验组数据
        control_data: 对照组数据
        metric_name: 指标名称
        variant_name: 实验组名称
        control_name: 对照组名称
        alpha: 显著性水平

    Returns:
        ABTestResult对象
    """
    # 基础统计
    n_variant = len(variant_data)
    n_control = len(control_data)
    mean_variant = variant_data.mean()
    mean_control = control_data.mean()

    # 提升度
    lift = (mean_variant - mean_control) / mean_control if mean_control != 0 else 0

    # t检验 (Welch's t-test，不假设方差相等)
    _, p_value = stats.ttest_ind(variant_data, control_data, equal_var=False)

    # 效应量
    effect_size = calculate_cohens_d(variant_data, control_data)

    # 统计功效
    power = calculate_statistical_power(abs(effect_size), n_variant, n_control, alpha)

    return ABTestResult(
        metric_name=metric_name,
        variant=variant_name,
        control=control_name,
        sample_size_variant=n_variant,
        sample_size_control=n_control,
        mean_variant=mean_variant,
        mean_control=mean_control,
        lift=lift,
        p_value=p_value,
        is_significant=p_value < alpha,
        confidence_level=1 - alpha,
        effect_size=effect_size,
        power=power
    )


def perform_mann_whitney_test(
    variant_data: pd.Series,
    control_data: pd.Series
) -> Tuple[float, float]:
    """
    执行Mann-Whitney U检验（非参数检验）
    适用于数据不满足正态分布假设的情况

    Args:
        variant_data: 实验组数据
        control_data: 对照组数据

    Returns:
        (U统计量, p值)
    """
    stat, p_value = stats.mannwhitneyu(
        variant_data, control_data, alternative='two-sided'
    )
    return stat, p_value


def calculate_bootstrap_ci(
    data: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    使用Bootstrap方法计算置信区间

    Args:
        data: 数据序列
        n_bootstrap: Bootstrap采样次数
        confidence: 置信水平

    Returns:
        (下界, 上界)
    """
    bootstrap_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = data.sample(n=n, replace=True)
        bootstrap_means.append(sample.mean())

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)

    return lower, upper


def analyze_metric_by_variant(
    df: pd.DataFrame,
    metric_col: str,
    variant_col: str = 'variant_list',
    control_variant: str = 'V2',
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对指定指标按实验分组进行完整的AB测试分析

    Args:
        df: 包含指标和分组的DataFrame
        metric_col: 指标列名
        variant_col: 分组列名
        control_variant: 对照组名称
        alpha: 显著性水平

    Returns:
        (描述性统计DataFrame, AB测试结果DataFrame)
    """
    variants = df[variant_col].unique()

    # 1. 描述性统计
    desc_stats_list = []
    for variant in sorted(variants):
        variant_data = df[df[variant_col] == variant][metric_col]
        stats_dict = calculate_descriptive_stats(variant_data)
        stats_dict['分组'] = variant
        stats_dict['指标'] = metric_col
        desc_stats_list.append(stats_dict)

    desc_stats_df = pd.DataFrame(desc_stats_list)
    # 调整列顺序
    cols = ['指标', '分组', '样本数', '均值', '标准差', '方差', '最小值', '最大值', 'P50', 'P90', 'P95', 'P99']
    desc_stats_df = desc_stats_df[cols]

    # 2. AB测试（各组 vs 对照组）
    control_data = df[df[variant_col] == control_variant][metric_col]

    ab_results_list = []
    for variant in sorted(variants):
        if variant == control_variant:
            continue

        variant_data = df[df[variant_col] == variant][metric_col]

        # 检查样本量是否足够（至少需要2个样本才能计算方差）
        if len(variant_data) < 2 or len(control_data) < 2:
            continue

        # 参数检验（t检验）
        ab_result = perform_ab_test(
            variant_data, control_data,
            metric_col, variant, control_variant, alpha
        )

        # 非参数检验（Mann-Whitney）
        try:
            _, mw_p = perform_mann_whitney_test(variant_data, control_data)
        except ValueError:
            mw_p = float('nan')

        # Bootstrap置信区间
        variant_ci = calculate_bootstrap_ci(variant_data)
        control_ci = calculate_bootstrap_ci(control_data)

        ab_results_list.append({
            '指标': metric_col,
            '实验组': variant,
            '对照组': control_variant,
            '实验组样本数': ab_result.sample_size_variant,
            '对照组样本数': ab_result.sample_size_control,
            '实验组均值': ab_result.mean_variant,
            '对照组均值': ab_result.mean_control,
            '提升度': ab_result.lift,
            '提升度(%)': f"{ab_result.lift * 100:.2f}%",
            't检验p值': ab_result.p_value,
            'Mann-Whitney p值': mw_p,
            '是否显著(α=0.05)': '是' if ab_result.is_significant else '否',
            "Cohen's d": ab_result.effect_size,
            '效应量解读': interpret_cohens_d(ab_result.effect_size),
            '统计功效': ab_result.power,
            '功效解读': '充足' if ab_result.power >= 0.8 else '不足',
            '实验组95%CI': f"[{variant_ci[0]:.4f}, {variant_ci[1]:.4f}]",
            '对照组95%CI': f"[{control_ci[0]:.4f}, {control_ci[1]:.4f}]",
        })

    ab_results_df = pd.DataFrame(ab_results_list)

    return desc_stats_df, ab_results_df


def interpret_cohens_d(d: float) -> str:
    """
    解读Cohen's d效应量

    Args:
        d: Cohen's d值

    Returns:
        效应量解读文字
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return '可忽略'
    elif d_abs < 0.5:
        return '小效应'
    elif d_abs < 0.8:
        return '中效应'
    else:
        return '大效应'


def run_comprehensive_ab_analysis(
    metrics_config: List[Dict],
    control_variant: str = 'V2',
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对多个指标进行综合AB测试分析

    Args:
        metrics_config: 指标配置列表，每项包含:
            - df: DataFrame
            - metric_col: 指标列名
            - variant_col: 分组列名（可选，默认'variant_list'）
        control_variant: 对照组名称
        alpha: 显著性水平

    Returns:
        (汇总描述性统计DataFrame, 汇总AB测试结果DataFrame)
    """
    all_desc_stats = []
    all_ab_results = []

    for config in metrics_config:
        df = config['df']
        metric_col = config['metric_col']
        variant_col = config.get('variant_col', 'variant_list')

        desc_stats, ab_results = analyze_metric_by_variant(
            df, metric_col, variant_col, control_variant, alpha
        )

        all_desc_stats.append(desc_stats)
        all_ab_results.append(ab_results)

    combined_desc_stats = pd.concat(all_desc_stats, ignore_index=True)
    combined_ab_results = pd.concat(all_ab_results, ignore_index=True)

    return combined_desc_stats, combined_ab_results


def print_ab_analysis_report(
    desc_stats_df: pd.DataFrame,
    ab_results_df: pd.DataFrame,
    title: str = "AB实验分析报告"
):
    """
    打印格式化的AB分析报告

    Args:
        desc_stats_df: 描述性统计DataFrame
        ab_results_df: AB测试结果DataFrame
        title: 报告标题
    """
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)

    # 按指标分组打印
    for metric in desc_stats_df['指标'].unique():
        print(f"\n{'─' * 80}")
        print(f"【指标: {metric}】")
        print(f"{'─' * 80}")

        # 描述性统计
        print("\n📊 描述性统计:")
        metric_desc = desc_stats_df[desc_stats_df['指标'] == metric].copy()
        # 格式化数值列
        for col in ['均值', '标准差', '方差', '最小值', '最大值', 'P50', 'P90', 'P95', 'P99']:
            metric_desc[col] = metric_desc[col].apply(lambda x: f"{x:.4f}")
        print(metric_desc.to_string(index=False))

        # AB测试结果
        print("\n🔬 AB测试结果 (vs V2对照组):")
        metric_ab = ab_results_df[ab_results_df['指标'] == metric].copy()
        if not metric_ab.empty:
            # 选择关键列展示
            key_cols = ['实验组', '提升度(%)', 't检验p值', '是否显著(α=0.05)',
                       "Cohen's d", '效应量解读', '统计功效', '功效解读']
            print(metric_ab[key_cols].to_string(index=False))

            # 显著性结论
            print("\n📈 结论:")
            for _, row in metric_ab.iterrows():
                sig_text = "✅ 显著" if row['是否显著(α=0.05)'] == '是' else "❌ 不显著"
                direction = "↑ 提升" if float(row['提升度(%)'].replace('%', '')) > 0 else "↓ 下降"
                print(f"   {row['实验组']} vs V2: {row['提升度(%)']} {direction} | {sig_text} | {row['效应量解读']} | 功效{row['功效解读']}")

    print("\n" + "=" * 80)




# 执行AB实验分析

# 定义要分析的指标配置
# CTR计算方式：用户级别聚合后计算 sum(点击)/sum(曝光)，避免SKU级别异常数据的影响
metrics_to_analyze = [
    {'df': user_sku_view_serial, 'metric_col': 'SKU曝光次数'},
    {'df': user_sku_click_serial, 'metric_col': 'SKU点击次数'},
    {'df': user_sku_ctr_serial, 'metric_col': 'CTR'},
    {'df': user_avg_position_serial, 'metric_col': '平均点击位置'},
]

# 运行综合分析
desc_stats_summary, ab_results_summary = run_comprehensive_ab_analysis(
    metrics_to_analyze,
    control_variant='V2',
    alpha=0.05
)

# 打印分析报告
print_ab_analysis_report(desc_stats_summary, ab_results_summary,
                        title=f"分类页LTR重排序AB实验分析报告 ({START_DATE} ~ 今)")

# 显示完整结果表格
print("\n\n📋 完整描述性统计表:")
print(desc_stats_summary)

print("\n📋 完整AB测试结果表:")
print(ab_results_summary)


# ============================================================================
# 📊 分组AB实验分析 - 按区域和90天交易属性
# ============================================================================

def run_segmented_ab_analysis(
    metrics_config: List[Dict],
    segment_col: str,
    segment_name: str,
    control_variant: str = 'V2',
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对指定分组进行AB实验分析

    Args:
        metrics_config: 指标配置列表
        segment_col: 分组列名
        segment_name: 分组名称（用于显示）
        control_variant: 对照组名称
        alpha: 显著性水平

    Returns:
        (汇总描述性统计DataFrame, 汇总AB测试结果DataFrame)
    """
    all_desc_stats = []
    all_ab_results = []

    for config in metrics_config:
        df = config['df']
        metric_col = config['metric_col']
        variant_col = config.get('variant_col', 'variant_list')

        # 获取所有分组值
        segments = df[segment_col].unique()

        for segment in sorted(segments, key=lambda x: str(x)):
            segment_df = df[df[segment_col] == segment]

            if len(segment_df) < 10:  # 样本量太少，跳过
                continue

            # 检查对照组是否有足够样本
            control_df = segment_df[segment_df[variant_col] == control_variant]
            if len(control_df) < 2:  # 对照组样本太少，跳过
                continue

            # 检查是否至少有一个实验组有足够样本
            other_variants = segment_df[segment_df[variant_col] != control_variant][variant_col].unique()
            has_valid_variant = False
            for v in other_variants:
                if len(segment_df[segment_df[variant_col] == v]) >= 2:
                    has_valid_variant = True
                    break
            if not has_valid_variant:
                continue

            try:
                desc_stats, ab_results = analyze_metric_by_variant(
                    segment_df, metric_col, variant_col, control_variant, alpha
                )
            except Exception as e:
                print(f"⚠️ 跳过分组 {segment}（{metric_col}）: {str(e)[:50]}")
                continue

            # 添加分组信息
            desc_stats['分组维度'] = segment_name
            desc_stats['分组值'] = segment
            ab_results['分组维度'] = segment_name
            ab_results['分组值'] = segment

            all_desc_stats.append(desc_stats)
            all_ab_results.append(ab_results)

    if not all_desc_stats:
        return pd.DataFrame(), pd.DataFrame()

    combined_desc_stats = pd.concat(all_desc_stats, ignore_index=True)
    combined_ab_results = pd.concat(all_ab_results, ignore_index=True)

    return combined_desc_stats, combined_ab_results


def print_segmented_ab_report(
    ab_results_df: pd.DataFrame,
    segment_name: str,
    title: str = "分组AB实验分析报告"
):
    """
    打印分组AB实验分析报告

    Args:
        ab_results_df: AB测试结果DataFrame
        segment_name: 分组名称
        title: 报告标题
    """
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

    if ab_results_df.empty:
        print("无数据")
        return

    # 按分组值和指标分组打印
    for segment_val in sorted(ab_results_df['分组值'].unique()):
        print(f"\n{'─' * 100}")
        print(f"【{segment_name}: {segment_val}】")
        print(f"{'─' * 100}")

        segment_data = ab_results_df[ab_results_df['分组值'] == segment_val]

        for metric in segment_data['指标'].unique():
            metric_data = segment_data[segment_data['指标'] == metric]
            print(f"\n  📊 {metric}:")

            for _, row in metric_data.iterrows():
                sig_text = "✅" if row['是否显著(α=0.05)'] == '是' else "❌"
                lift_val = float(row['提升度(%)'].replace('%', ''))
                direction = "↑" if lift_val > 0 else "↓"
                print(f"     {row['实验组']} vs V2: {row['提升度(%)']} {direction} | {sig_text} | p={row['t检验p值']:.4f}")

    print("\n" + "=" * 100)


# ============================================================================
# 1. 按区域分组的AB实验分析
# ============================================================================
print("\n\n")
print("=" * 100)
print("  🌍 按区域分组的AB实验分析")
print("=" * 100)

region_desc_stats, region_ab_results = run_segmented_ab_analysis(
    metrics_to_analyze,
    segment_col='region',
    segment_name='区域',
    control_variant='V2',
    alpha=0.05
)

print_segmented_ab_report(region_ab_results, '区域',
                         title=f"分类页LTR重排序AB实验 - 区域分组分析 ({START_DATE} ~ 今)")

# 输出区域分组的汇总表格
if not region_ab_results.empty:
    print("\n📋 区域分组AB测试汇总表:")
    region_summary = region_ab_results[['分组值', '指标', '实验组', '实验组样本数', '对照组样本数',
                                         '提升度(%)', 't检验p值', '是否显著(α=0.05)']].copy()
    print(region_summary.to_string(index=False))


# ============================================================================
# 2. 按90天交易属性分组的AB实验分析
# ============================================================================
print("\n\n")
print("=" * 100)
print("  💰 按90天交易属性分组的AB实验分析")
print("=" * 100)

trd_desc_stats, trd_ab_results = run_segmented_ab_analysis(
    metrics_to_analyze,
    segment_col='has_trd_amt_90d',
    segment_name='90天交易属性',
    control_variant='V2',
    alpha=0.05
)

print_segmented_ab_report(trd_ab_results, '90天交易属性',
                         title=f"分类页LTR重排序AB实验 - 90天交易属性分组分析 ({START_DATE} ~ 今)")

# 输出90天交易属性分组的汇总表格
if not trd_ab_results.empty:
    print("\n📋 90天交易属性分组AB测试汇总表:")
    trd_summary = trd_ab_results[['分组值', '指标', '实验组', '实验组样本数', '对照组样本数',
                                   '提升度(%)', 't检验p值', '是否显著(α=0.05)']].copy()
    print(trd_summary.to_string(index=False))


# ============================================================================
# 3. 按一级类目分组的AB实验分析
# ============================================================================
print("\n\n")
print("=" * 100)
print("  📦 按一级类目分组的AB实验分析")
print("=" * 100)

# 定义按一级类目分组的指标配置
metrics_by_cate1 = [
    {'df': user_sku_view_by_cate1, 'metric_col': 'SKU曝光次数'},
    {'df': user_sku_click_by_cate1, 'metric_col': 'SKU点击次数'},
    {'df': user_ctr_by_cate1, 'metric_col': 'CTR'},
    {'df': user_avg_position_by_cate1, 'metric_col': '平均点击位置'},
]

cate1_desc_stats, cate1_ab_results = run_segmented_ab_analysis(
    metrics_by_cate1,
    segment_col='cate_level1_id',
    segment_name='一级类目',
    control_variant='V2',
    alpha=0.05
)

print_segmented_ab_report(cate1_ab_results, '一级类目',
                         title=f"分类页LTR重排序AB实验 - 一级类目分组分析 ({START_DATE} ~ 今)")

# 输出一级类目分组的汇总表格
if not cate1_ab_results.empty:
    print("\n📋 一级类目分组AB测试汇总表:")
    cate1_summary = cate1_ab_results[['分组值', '指标', '实验组', '实验组样本数', '对照组样本数',
                                       '提升度(%)', 't检验p值', '是否显著(α=0.05)']].copy()
    print(cate1_summary.to_string(index=False))


# ============================================================================
# 4. 按二级类目分组的AB实验分析
# ============================================================================
print("\n\n")
print("=" * 100)
print("  📦 按二级类目分组的AB实验分析")
print("=" * 100)

# 定义按二级类目分组的指标配置
metrics_by_cate2 = [
    {'df': user_sku_view_by_cate2, 'metric_col': 'SKU曝光次数'},
    {'df': user_sku_click_by_cate2, 'metric_col': 'SKU点击次数'},
    {'df': user_ctr_by_cate2, 'metric_col': 'CTR'},
    {'df': user_avg_position_by_cate2, 'metric_col': '平均点击位置'},
]

cate2_desc_stats, cate2_ab_results = run_segmented_ab_analysis(
    metrics_by_cate2,
    segment_col='cate_level2_id',
    segment_name='二级类目',
    control_variant='V2',
    alpha=0.05
)

print_segmented_ab_report(cate2_ab_results, '二级类目',
                         title=f"分类页LTR重排序AB实验 - 二级类目分组分析 ({START_DATE} ~ 今)")

# 输出二级类目分组的汇总表格
if not cate2_ab_results.empty:
    print("\n📋 二级类目分组AB测试汇总表:")
    cate2_summary = cate2_ab_results[['分组值', '指标', '实验组', '实验组样本数', '对照组样本数',
                                       '提升度(%)', 't检验p值', '是否显著(α=0.05)']].copy()
    print(cate2_summary.to_string(index=False))


# ============================================================================
# 📊 AB测试结果可视化模块 - 详细指标对比
# ============================================================================

def create_ab_test_visualization(
    desc_stats_df: pd.DataFrame,
    ab_results_df: pd.DataFrame,
    title_prefix: str = "分类页LTR重排序"
):
    """
    创建AB测试结果的详细可视化图表
    展示SKU曝光数、点击数、点击率、平均点击位置、加入购物车转化率等关键指标

    Args:
        desc_stats_df: 描述性统计DataFrame
        ab_results_df: AB测试结果DataFrame
        title_prefix: 图表标题前缀
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import sys

    # 先设置seaborn主题（会重置字体）
    sns.set_theme(style="whitegrid", palette="husl")
    sns.set_palette("husl")

    # 然后重新设置中文字体（必须在sns.set_theme之后）
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'STHeiti', 'Songti SC', 'Kaiti SC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.family'] = 'sans-serif'

    # 获取所有指标
    metrics = ab_results_df['指标'].unique()

    # 为每个指标创建对比可视化
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'{title_prefix}AB实验详细指标对比 (V1/V3/V4 vs V2)',
                 fontsize=18, fontweight='bold', y=0.995)

    # 颜色方案
    colors = {'V1': '#3498db', 'V2': '#e74c3c', 'V3': '#27ae60', 'V4': '#9b59b6'}

    plot_idx = 1
    for metric in sorted(metrics):
        # 获取当前指标的描述性统计
        metric_desc = desc_stats_df[desc_stats_df['指标'] == metric].copy()
        metric_ab = ab_results_df[ab_results_df['指标'] == metric].copy()

        if metric_desc.empty:
            continue

        # 子图1: 各分组均值对比
        ax = fig.add_subplot(len(metrics), 3, plot_idx)
        plot_idx += 1

        variants = sorted(metric_desc['分组'].unique())
        means = [metric_desc[metric_desc['分组'] == v]['均值'].values[0] for v in variants]
        stds = [metric_desc[metric_desc['分组'] == v]['标准差'].values[0] for v in variants]

        bars = ax.bar(variants, means, yerr=stds, capsize=5, alpha=0.8,
                     color=[colors.get(v, '#95a5a6') for v in variants],
                     edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(f'{metric} - 均值对比', fontsize=12, fontweight='bold')
        ax.set_ylabel('均值', fontsize=10)
        ax.set_ylim(0, max(means) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 子图2: 样本数对比
        ax = fig.add_subplot(len(metrics), 3, plot_idx)
        plot_idx += 1

        sample_sizes = [metric_desc[metric_desc['分组'] == v]['样本数'].values[0] for v in variants]
        bars = ax.bar(variants, sample_sizes, alpha=0.8,
                     color=[colors.get(v, '#95a5a6') for v in variants],
                     edgecolor='black', linewidth=1.5)

        for bar, size in zip(bars, sample_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(size):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'{metric} - 样本数', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本数', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 子图3: 提升度和显著性
        ax = fig.add_subplot(len(metrics), 3, plot_idx)
        plot_idx += 1

        if not metric_ab.empty:
            exp_variants = metric_ab['实验组'].values
            lifts = [float(row['提升度(%)'].replace('%', '')) for _, row in metric_ab.iterrows()]
            is_sig = [row['是否显著(α=0.05)'] == '是' for _, row in metric_ab.iterrows()]

            # 按提升度大小排序
            sorted_indices = np.argsort(lifts)[::-1]
            exp_variants = exp_variants[sorted_indices]
            lifts = np.array(lifts)[sorted_indices]
            is_sig = np.array(is_sig)[sorted_indices]

            # 根据显著性着色
            bar_colors = ['#27ae60' if sig and lift > 0 else '#c0392b' if sig and lift < 0
                         else '#f39c12' if not sig and lift > 0 else '#bdc3c7'
                         for sig, lift in zip(is_sig, lifts)]

            bars = ax.barh(exp_variants, lifts, alpha=0.85, color=bar_colors, edgecolor='black', linewidth=1.5)

            # 添加数值和显著性标记
            for bar, lift, sig in zip(bars, lifts, is_sig):
                width = bar.get_width()
                sig_mark = '*' if sig else ''  # 显著用*标记，不显著无标记
                ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2.,
                       f'{lift:+.2f}% {sig_mark}', ha='left' if width > 0 else 'right',
                       va='center', fontsize=10, fontweight='bold')

            ax.axvline(x=0, color='#e74c3c', linestyle='-', linewidth=2)
            ax.set_title(f'{metric} - 提升度 (vs V2)', fontsize=12, fontweight='bold')
            ax.set_xlabel('提升度 (%)', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('./data/ab_test_metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 详细指标对比图表已保存到: ./data/ab_test_metrics_comparison.png")
    plt.show()


# 执行可视化
create_ab_test_visualization(desc_stats_summary, ab_results_summary,
                            title_prefix=f"分类页LTR重排序 ({START_DATE})")


# ============================================================================
# 📊 分组AB实验结果可视化
# ============================================================================

def create_segmented_ab_visualization(
    ab_results_df: pd.DataFrame,
    segment_name: str,
    title_prefix: str = "分类页LTR重排序",
    user_count_df: pd.DataFrame = None
):
    """
    创建分组AB测试结果的可视化图表
    展示不同分组下各实验变体相对于对照组的提升度

    Args:
        ab_results_df: AB测试结果DataFrame
        segment_name: 分组名称
        title_prefix: 图表标题前缀
        user_count_df: 用户数量统计DataFrame，包含分组值和用户数
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import sys

    if ab_results_df.empty:
        print(f"⚠️ {segment_name}分组无数据，跳过可视化")
        return

    # 设置seaborn主题
    sns.set_theme(style="whitegrid", palette="husl")

    # 设置中文字体
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'STHeiti', 'Songti SC', 'Kaiti SC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 获取指标列表和分组值列表
    metrics = ab_results_df['指标'].unique()
    segments = sorted(ab_results_df['分组值'].unique())
    variants = sorted(ab_results_df['实验组'].unique())

    # 构建分组标签（包含用户数）
    segment_labels = []
    for seg in segments:
        if user_count_df is not None and seg in user_count_df.index:
            user_count = user_count_df.loc[seg]
            segment_labels.append(f"{seg}\n(n={user_count:,})")
        else:
            segment_labels.append(seg)

    # 创建图表
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(f'{title_prefix} - {segment_name}分组AB实验提升度对比',
                 fontsize=16, fontweight='bold', y=0.995)

    # 颜色方案
    colors = {'V1': '#3498db', 'V3': '#27ae60', 'V4': '#9b59b6'}
    segment_positions = np.arange(len(segments))
    bar_width = 0.25

    for idx, metric in enumerate(sorted(metrics)):
        ax = axes[idx]
        metric_data = ab_results_df[ab_results_df['指标'] == metric]

        for i, variant in enumerate(variants):
            variant_data = metric_data[metric_data['实验组'] == variant]

            lifts = []
            is_sigs = []
            for segment in segments:
                seg_data = variant_data[variant_data['分组值'] == segment]
                if not seg_data.empty:
                    lift_str = seg_data['提升度(%)'].values[0]
                    lifts.append(float(lift_str.replace('%', '')))
                    is_sigs.append(seg_data['是否显著(α=0.05)'].values[0] == '是')
                else:
                    lifts.append(0)
                    is_sigs.append(False)

            # 绘制柱状图
            positions = segment_positions + (i - 1) * bar_width
            bars = ax.bar(positions, lifts, bar_width, label=variant,
                         color=colors.get(variant, '#95a5a6'), alpha=0.85,
                         edgecolor='black', linewidth=1)

            # 添加数值标签和显著性标记
            for bar, lift, is_sig in zip(bars, lifts, is_sigs):
                height = bar.get_height()
                sig_mark = '*' if is_sig else ''
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.5 if height >= 0 else -0.5
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{lift:+.1f}%{sig_mark}', ha='center', va=va,
                       fontsize=8, fontweight='bold')

        ax.set_ylabel('提升度 (%)', fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(segment_positions)
        ax.set_xticklabels(segment_labels, fontsize=10)  # 使用包含用户数的标签
        ax.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=1.5)
        ax.legend(loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = f'./data/ab_test_{segment_name}_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ {segment_name}分组对比图表已保存到: {filename}")
    plt.show()


# 执行分组可视化
# 计算区域分组的用户数统计
region_user_counts = user_click_with_variant_statistics_df.groupby('region')['uid'].nunique()
print("\n📊 区域分组用户数统计:")
print(region_user_counts)

# 计算90天交易属性分组的用户数统计
trd_user_counts = user_click_with_variant_statistics_df.groupby('has_trd_amt_90d')['uid'].nunique()
print("\n📊 90天交易属性分组用户数统计:")
print(trd_user_counts)

if not region_ab_results.empty:
    create_segmented_ab_visualization(region_ab_results, '区域',
                                      title_prefix=f"分类页LTR重排序 ({START_DATE})",
                                      user_count_df=region_user_counts)

if not trd_ab_results.empty:
    create_segmented_ab_visualization(trd_ab_results, '90天交易属性',
                                      title_prefix=f"分类页LTR重排序 ({START_DATE})",
                                      user_count_df=trd_user_counts)

# 计算一级类目分组的用户数统计
cate1_user_counts = user_click_with_variant_statistics_df.groupby('cate_level1_id')['uid'].nunique()
print("\n📊 一级类目分组用户数统计:")
print(cate1_user_counts.sort_values(ascending=False).head(20))

# 计算二级类目分组的用户数统计
cate2_user_counts = user_click_with_variant_statistics_df.groupby('cate_level2_id')['uid'].nunique()
print("\n📊 二级类目分组用户数统计 (Top 20):")
print(cate2_user_counts.sort_values(ascending=False).head(20))

if not cate1_ab_results.empty:
    create_segmented_ab_visualization(cate1_ab_results, '一级类目',
                                      title_prefix=f"分类页LTR重排序 ({START_DATE})",
                                      user_count_df=cate1_user_counts)

if not cate2_ab_results.empty:
    # 二级类目可能较多，只可视化用户数Top10的类目
    top_cate2 = cate2_user_counts.sort_values(ascending=False).head(10).index.tolist()
    cate2_ab_results_top = cate2_ab_results[cate2_ab_results['分组值'].isin(top_cate2)]
    if not cate2_ab_results_top.empty:
        create_segmented_ab_visualization(cate2_ab_results_top, '二级类目Top10',
                                          title_prefix=f"分类页LTR重排序 ({START_DATE})",
                                          user_count_df=cate2_user_counts)


# 将AB测试结果写入本地CSV文件
# 获取数据的开始和结束日期
data_start_date = all_user_variant_df['ds'].min() if not all_user_variant_df.empty else START_DATE.replace('-', '')
data_end_date = all_user_variant_df['ds'].max() if not all_user_variant_df.empty else datetime.now().strftime('%Y%m%d')

# 构建CSV文件名，包含开始和结束日期
csv_filename = f"./data/ab_test_results_{data_start_date}_{data_end_date}.csv"

# 写入CSV文件
ab_results_summary.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n✅ AB测试结果已保存到: {csv_filename}")

# 保存分组分析结果到CSV
if not region_ab_results.empty:
    region_csv_filename = f"./data/ab_test_results_by_region_{data_start_date}_{data_end_date}.csv"
    region_ab_results.to_csv(region_csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 区域分组AB测试结果已保存到: {region_csv_filename}")

if not trd_ab_results.empty:
    trd_csv_filename = f"./data/ab_test_results_by_trd_amt_{data_start_date}_{data_end_date}.csv"
    trd_ab_results.to_csv(trd_csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 90天交易属性分组AB测试结果已保存到: {trd_csv_filename}")

if not cate1_ab_results.empty:
    cate1_csv_filename = f"./data/ab_test_results_by_cate1_{data_start_date}_{data_end_date}.csv"
    cate1_ab_results.to_csv(cate1_csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 一级类目分组AB测试结果已保存到: {cate1_csv_filename}")

if not cate2_ab_results.empty:
    cate2_csv_filename = f"./data/ab_test_results_by_cate2_{data_start_date}_{data_end_date}.csv"
    cate2_ab_results.to_csv(cate2_csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 二级类目分组AB测试结果已保存到: {cate2_csv_filename}")

if not args.upload_odps:
    print("未指定上传到ODPS，程序结束")


# ## 以下是ODPS分析（仅在指定--upload-odps参数时执行）



if args.upload_odps:
    from odps_client import write_pandas_df_into_odps
    from datetime import datetime
    import numpy as np

    # 在上传ODPS前，清理所有列中的'null'字符串
    df_to_upload = user_click_with_variant_statistics_df.copy()

    print(f"📊 原始数据行数: {len(df_to_upload)}")
    print(f"📊 原始数据列: {df_to_upload.columns.tolist()}")
    print(f"📊 原始数据类型:\n{df_to_upload.dtypes}")

    # 第一步：将所有列中的'null'字符串替换为真正的NaN
    df_to_upload = df_to_upload.replace('null', np.nan)
    df_to_upload = df_to_upload.replace('NULL', np.nan)
    df_to_upload = df_to_upload.replace('None', np.nan)

    # 第二步：处理所有列，尝试转换为数值类型
    for col in df_to_upload.columns:
        # 跳过明确的字符串列
        if col in ['uid', 'variant_list', 'ds', 'sku_id', 'category_id']:
            continue
        # 尝试将列转换为数值类型
        df_to_upload[col] = pd.to_numeric(df_to_upload[col], errors='ignore')

    # 第三步：检查是否还有'null'字符串
    for col in df_to_upload.columns:
        if df_to_upload[col].dtype == 'object':
            null_count = df_to_upload[col].astype(str).str.lower().eq('null').sum()
            if null_count > 0:
                print(f"⚠️ 列 {col} 仍包含 {null_count} 个 'null' 字符串，进行清理...")
                df_to_upload[col] = df_to_upload[col].replace(
                    to_replace=r'(?i)^null$', value=np.nan, regex=True
                )

    # 移除完全为NaN的行
    df_to_upload = df_to_upload.dropna(how='all')

    print(f"📊 清理后数据类型:\n{df_to_upload.dtypes}")
    print(f"📊 数据清理完成，清理后行数: {len(df_to_upload)}")

    table_name = "summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df"
    partition_spec = f"ds={datetime.now().strftime('%Y%m%d')}"
    write_pandas_df_into_odps(
        df=df_to_upload,
        table_name=table_name,
        partition_spec=partition_spec,
        overwrite=True,
        lifecycle=30,
    )
    print(f"数据已成功写入ODPS表: {table_name}，分区: {partition_spec}")
    print(f"数据行数: {len(df_to_upload)}")
    print("\n✅ ODPS数据上传完成！")

