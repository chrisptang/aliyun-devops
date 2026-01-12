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
    '/goods' as "页面名称",uid,sku_viewed_or_clicked,
    count(distinct case when type='view' and position_id='goods' then (uid,sku_viewed_or_clicked,__time__,idx) end) as "SKU曝光次数",
    count(distinct case when type='cl' and position_id in ('goods','唤起购买') then (uid,sku_viewed_or_clicked,__time__,idx) end) as "SKU点击次数",
    round(1.00*count(distinct case when type='cl' and position_id in ('goods','唤起购买') then (uid,sku_viewed_or_clicked,__time__,idx) end)/count(distinct case when type='view' and position_id='goods' then (uid,sku_viewed_or_clicked,__time__,idx) end),4) as "SKU点击率",
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
group by 1,2,3,4
having length(uid)>0
order by 5 desc
"""


def get_user_sku_view_of_date_from_sls(
    day: datetime, check_if_local_exist: bool = True
) -> pd.DataFrame:
    db_file_name = f"./data/category_ltr_ab_user_sku_view.db"
    table_name = f"user_sku_view_{day.strftime('%Y%m%d')}"
    conn = sqlite3.connect(db_file_name)
    cursor = conn.cursor()

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

# 聚合1: SKU曝光次数
user_sku_view_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({"SKU曝光次数": "sum"})
    .reset_index()
)

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

user_avg_position_serial.head(5)

# 聚合3: SKU点击次数
user_sku_click_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({"SKU点击次数": "sum"})
    .reset_index()
)

user_sku_click_serial.head(5)




# 聚合4: CTR计算
user_sku_ctr_serial = (
    user_click_with_variant_statistics_df
    .groupby(["variant_list", "uid"])
    .agg({
        "SKU点击次数": "sum",
        "SKU曝光次数": "sum"
    })
    .reset_index()
)

# 过滤掉曝光次数为0的记录（避免除以零产生inf）
user_sku_ctr_serial = user_sku_ctr_serial[user_sku_ctr_serial['SKU曝光次数'] > 0].copy()

# 计算CTR
user_sku_ctr_serial['CTR'] = user_sku_ctr_serial['SKU点击次数'] / user_sku_ctr_serial['SKU曝光次数']

# 处理异常CTR值（理论上CTR应该在0-1之间，超过1的是数据异常）
user_sku_ctr_serial['CTR'] = user_sku_ctr_serial['CTR'].clip(upper=1.0)

user_sku_ctr_serial.head(5)


# ## AB实验统计分析模块



import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
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
    t_stat, p_value = stats.ttest_ind(variant_data, control_data, equal_var=False)

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

        # 参数检验（t检验）
        ab_result = perform_ab_test(
            variant_data, control_data,
            metric_col, variant, control_variant, alpha
        )

        # 非参数检验（Mann-Whitney）
        mw_stat, mw_p = perform_mann_whitney_test(variant_data, control_data)

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

if not args.upload_odps:
    print("未指定上传到ODPS，程序结束")


# ## 以下是ODPS分析（仅在指定--upload-odps参数时执行）



if args.upload_odps:
    from odps_client import get_odps_sql_result_as_df, write_pandas_df_into_odps
    from datetime import datetime

    table_name = "summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df"
    partition_spec = f"ds={datetime.now().strftime('%Y%m%d')}"
    write_pandas_df_into_odps(
        df=user_click_with_variant_statistics_df,
        table_name=table_name,
        partition_spec=partition_spec,
        overwrite=True,
        lifecycle=30,
    )
    print(f"数据已成功写入ODPS表: {table_name}，分区: {partition_spec}")
    print(f"数据行数: {len(user_click_with_variant_statistics_df)}")

if args.upload_odps:
    from odps_client import get_odps_sql_result_as_df
    from datetime import datetime,timedelta

    DAYS_TO_BE_NEW_SKU=3650
    order_date_to_be_new_sku=(datetime.now()-timedelta(days=DAYS_TO_BE_NEW_SKU)).strftime("%Y%m%d")

    sku_view_sql=f"""
    SELECT  expe.variant_list
            ,CASE   WHEN old.viewed_cnt > 0 THEN '之前购买过的SKU'
                    ELSE '新SKU'
            END AS 是否新SKU
            ,COUNT(DISTINCT expe.uid) 用户数
            ,SUM(sku点击总次数) sku点击总次数
            ,SUM(sku曝光总次数) sku曝光总次数
            ,SUM(加入购物车立即购买总次数) 加入购物车立即购买总次数
    FROM    (
                SELECT  uid
                        ,variant_list
                        ,sku_viewed_or_clicked
                        ,SUM(COALESCE(sku点击次数,0)) sku点击总次数
                        ,SUM(sku曝光次数) sku曝光总次数
                        ,SUM(COALESCE(加入购物车次数,0)+COALESCE(立即购买次数,0)) 加入购物车立即购买总次数
                FROM    {table_name}
                WHERE   ds = MAX_PT('{table_name}')
                GROUP BY uid
                         ,variant_list
                         ,sku_viewed_or_clicked
            ) expe
    LEFT JOIN   (
                    SELECT  cust_id
                            ,sku_id
                            ,COUNT(*) viewed_cnt
                    FROM    summerfarm_tech.dwd_trd_order_df
                    WHERE   ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
                    AND     order_date >= '{order_date_to_be_new_sku}'
                    GROUP BY cust_id
                             ,sku_id
                ) old
    ON      expe.uid = old.cust_id
    AND     expe.sku_viewed_or_clicked = old.sku_id
    GROUP BY expe.variant_list
             ,CASE   WHEN old.viewed_cnt > 0 THEN '之前购买过的SKU'
                     ELSE '新SKU'
             END
    order by expe.variant_list
             ,CASE   WHEN old.viewed_cnt > 0 THEN '之前购买过的SKU'
                     ELSE '新SKU'
             END;
    """

    result_df=get_odps_sql_result_as_df(sku_view_sql)
    result_df





    result_df['SKU点击率']=result_df['sku点击总次数']/result_df['sku曝光总次数']
    variant_group=result_df.groupby('variant_list').agg({'sku点击总次数':'sum','sku曝光总次数':'sum','加入购物车立即购买总次数':'sum'}).reset_index()
    variant_group['分组SKU点击率']=variant_group['sku点击总次数']/variant_group['sku曝光总次数']
    variant_group['分组加入购物车立即购买率']=variant_group['加入购物车立即购买总次数']/variant_group['sku曝光总次数']

    variant_group.columns=['variant_list', '分组sku点击总次数','分组sku曝光总次数','分组加入购物车立即购买总次数','分组SKU点击率','分组加入购物车立即购买率']

    v2_ctr=variant_group[variant_group['variant_list']=='V2']['分组SKU点击率'].values[0]
    v2_jg=variant_group[variant_group['variant_list']=='V2']['分组加入购物车立即购买率'].values[0]
    result_final_df=result_df.merge(variant_group,on='variant_list',how='left')
    result_final_df['V2点击率对比']=result_final_df['分组SKU点击率']/v2_ctr-1.00
    result_final_df['V2加入购物车立即购买率对比']=result_final_df['分组加入购物车立即购买率']/v2_jg-1.00
    result_final_df





    print(result_final_df.columns)
    date_range = user_click_with_variant_statistics_df['ds'].min() + '~' + user_click_with_variant_statistics_df['ds'].max()

    result_final_df





    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter, FuncFormatter

    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置seaborn专业风格
    sns.set_theme(style="whitegrid", font='PingFang SC', palette="husl")
    sns.set_context("talk", font_scale=0.9)

    # 确保result_final_df存在
    print("数据列:", result_final_df.columns.tolist())
    print("\n原始数据:")
    result_final_df





    # ================================================================================
    # 推荐系统AB实验分析 - V1/V3/V4 vs V2(对照组) 核心指标专业对比图
    # ================================================================================

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    import warnings
    warnings.filterwarnings('ignore')

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font='Arial Unicode MS')

    # ==================== 数据准备 ====================
    df = result_final_df.copy()

    # 分组级别数据（去重）
    df_group = df[['variant_list', '分组sku点击总次数', '分组sku曝光总次数', '分组加入购物车立即购买总次数',
                   '分组SKU点击率', '分组加入购物车立即购买率', 'V2点击率对比', 'V2加入购物车立即购买率对比']].drop_duplicates().reset_index(drop=True)

    # 新SKU数据计算
    df_new_sku = df[df['是否新sku'] == '新SKU'][['variant_list', 'sku点击总次数']].copy()
    df_new_sku.columns = ['variant_list', '新SKU点击次数']
    df_total = df.groupby('variant_list')['sku点击总次数'].sum().reset_index()
    df_total.columns = ['variant_list', '总点击次数']
    df_new_ratio = df_new_sku.merge(df_total, on='variant_list')
    df_new_ratio['新SKU点击占比'] = df_new_ratio['新SKU点击次数'] / df_new_ratio['总点击次数']

    # V2基准值
    v2_new_ratio_val = df_new_ratio[df_new_ratio['variant_list'] == 'V2']['新SKU点击占比'].values[0]
    df_new_ratio['新SKU点击占比_vs_V2'] = (df_new_ratio['新SKU点击占比'] - v2_new_ratio_val) / v2_new_ratio_val

    # 合并数据
    df_plot = df_group.merge(df_new_ratio[['variant_list', '新SKU点击次数', '新SKU点击占比', '新SKU点击占比_vs_V2']], on='variant_list')

    # V2基准值
    v2_clicks = df_plot[df_plot['variant_list'] == 'V2']['分组sku点击总次数'].values[0]
    v2_exposure = df_plot[df_plot['variant_list'] == 'V2']['分组sku曝光总次数'].values[0]
    v2_cart = df_plot[df_plot['variant_list'] == 'V2']['分组加入购物车立即购买总次数'].values[0]
    v2_ctr = df_plot[df_plot['variant_list'] == 'V2']['分组SKU点击率'].values[0]
    v2_cvr = df_plot[df_plot['variant_list'] == 'V2']['分组加入购物车立即购买率'].values[0]
    v2_new_click = df_plot[df_plot['variant_list'] == 'V2']['新SKU点击次数'].values[0]

    # 颜色方案 - V2红色(对照组)，其他蓝绿紫色系
    colors = {'V1': '#3498db', 'V2': '#e74c3c', 'V3': '#27ae60', 'V4': '#9b59b6'}

    # ==================== 创建综合图表 ====================
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(f'推荐系统AB实验分析 - V1/V3/V4 vs V2(对照组) 核心指标对比(新SKU定义:{DAYS_TO_BE_NEW_SKU}未购买) {date_range}', fontsize=22, fontweight='bold', y=0.98)

    # ---------- 图1: 分组SKU点击总次数 ----------
    ax1 = fig.add_subplot(3, 3, 1)
    bars1 = ax1.bar(df_plot['variant_list'], df_plot['分组sku点击总次数'],
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars1, df_plot.iterrows()):
        height = bar.get_height()
        diff = (row['分组sku点击总次数'] - v2_clicks) / v2_clicks * 100
        label = f'{int(height):,}\n(基准)' if row['variant_list'] == 'V2' else f'{int(height):,}\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax1.text(bar.get_x() + bar.get_width()/2., height + 30, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax1.set_title('分组SKU点击总次数', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('点击次数', fontsize=12)
    ax1.set_ylim(0, max(df_plot['分组sku点击总次数']) * 1.18)
    ax1.axhline(y=v2_clicks, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ---------- 图2: 分组SKU曝光总次数 ----------
    ax2 = fig.add_subplot(3, 3, 2)
    bars2 = ax2.bar(df_plot['variant_list'], df_plot['分组sku曝光总次数'],
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars2, df_plot.iterrows()):
        height = bar.get_height()
        diff = (row['分组sku曝光总次数'] - v2_exposure) / v2_exposure * 100
        label = f'{int(height):,}\n(基准)' if row['variant_list'] == 'V2' else f'{int(height):,}\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax2.text(bar.get_x() + bar.get_width()/2., height + 600, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax2.set_title('分组SKU曝光总次数', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('曝光次数', fontsize=12)
    ax2.set_ylim(0, max(df_plot['分组sku曝光总次数']) * 1.18)
    ax2.axhline(y=v2_exposure, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ---------- 图3: 分组加入购物车立即购买总次数 ----------
    ax3 = fig.add_subplot(3, 3, 3)
    bars3 = ax3.bar(df_plot['variant_list'], df_plot['分组加入购物车立即购买总次数'],
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars3, df_plot.iterrows()):
        height = bar.get_height()
        diff = (row['分组加入购物车立即购买总次数'] - v2_cart) / v2_cart * 100
        label = f'{int(height):,}\n(基准)' if row['variant_list'] == 'V2' else f'{int(height):,}\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax3.text(bar.get_x() + bar.get_width()/2., height + 8, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax3.set_title('分组加入购物车立即购买总次数', fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel('次数', fontsize=12)
    ax3.set_ylim(0, max(df_plot['分组加入购物车立即购买总次数']) * 1.18)
    ax3.axhline(y=v2_cart, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ---------- 图4: 新SKU点击次数 ----------
    ax4 = fig.add_subplot(3, 3, 4)
    bars4 = ax4.bar(df_plot['variant_list'], df_plot['新SKU点击次数'],
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars4, df_plot.iterrows()):
        height = bar.get_height()
        diff = (row['新SKU点击次数'] - v2_new_click) / v2_new_click * 100
        label = f'{int(height):,}\n(基准)' if row['variant_list'] == 'V2' else f'{int(height):,}\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax4.text(bar.get_x() + bar.get_width()/2., height + 15, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax4.set_title('新SKU点击总次数', fontsize=14, fontweight='bold', pad=15)
    ax4.set_ylabel('点击次数', fontsize=12)
    ax4.set_ylim(0, max(df_plot['新SKU点击次数']) * 1.18)
    ax4.axhline(y=v2_new_click, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # ---------- 图5: 新SKU点击占比 (及相比V2提升) ----------
    ax5 = fig.add_subplot(3, 3, 5)
    bars5 = ax5.bar(df_plot['variant_list'], df_plot['新SKU点击占比'] * 100,
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars5, df_plot.iterrows()):
        height = bar.get_height()
        diff = row['新SKU点击占比_vs_V2'] * 100
        label = f'{height:.1f}%\n(基准)' if row['variant_list'] == 'V2' else f'{height:.1f}%\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax5.set_title('新SKU点击占比 (及相比V2提升)', fontsize=14, fontweight='bold', pad=15)
    ax5.set_ylabel('占比 (%)', fontsize=12)
    ax5.set_ylim(0, max(df_plot['新SKU点击占比'] * 100) * 1.18)
    ax5.axhline(y=v2_new_ratio_val * 100, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # ---------- 图6: V2点击率对比 (分组SKU点击率) ----------
    ax6 = fig.add_subplot(3, 3, 6)
    bars6 = ax6.bar(df_plot['variant_list'], df_plot['分组SKU点击率'] * 100,
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars6, df_plot.iterrows()):
        height = bar.get_height()
        diff = row['V2点击率对比'] * 100
        label = f'{height:.2f}%\n(基准)' if row['variant_list'] == 'V2' else f'{height:.2f}%\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.03, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax6.set_title('V2点击率对比 (分组SKU点击率)', fontsize=14, fontweight='bold', pad=15)
    ax6.set_ylabel('点击率 (%)', fontsize=12)
    ax6.set_ylim(0, max(df_plot['分组SKU点击率'] * 100) * 1.20)
    ax6.axhline(y=v2_ctr * 100, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # ---------- 图7: V2加入购物车立即购买率对比 ----------
    ax7 = fig.add_subplot(3, 3, 7)
    bars7 = ax7.bar(df_plot['variant_list'], df_plot['分组加入购物车立即购买率'] * 100,
                    color=[colors[v] for v in df_plot['variant_list']], edgecolor='white', linewidth=2, alpha=0.85)
    for bar, (_, row) in zip(bars7, df_plot.iterrows()):
        height = bar.get_height()
        diff = row['V2加入购物车立即购买率对比'] * 100
        label = f'{height:.3f}%\n(基准)' if row['variant_list'] == 'V2' else f'{height:.3f}%\n({diff:+.1f}%)'
        color = '#e74c3c' if row['variant_list'] == 'V2' else ('#27ae60' if diff > 0 else '#c0392b')
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.0008, label, ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    ax7.set_title('V2加入购物车立即购买率对比', fontsize=14, fontweight='bold', pad=15)
    ax7.set_ylabel('转化率 (%)', fontsize=12)
    ax7.set_ylim(0, max(df_plot['分组加入购物车立即购买率'] * 100) * 1.22)
    ax7.axhline(y=v2_cvr * 100, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # ---------- 图8: 各实验组 vs V2 综合变化率对比 ----------
    ax8 = fig.add_subplot(3, 3, 8)
    metrics = ['点击次数', '曝光次数', '加购次数', '新SKU点击', '新SKU占比', '点击率', '转化率']
    v1_changes = [
        (df_plot[df_plot['variant_list']=='V1']['分组sku点击总次数'].values[0] - v2_clicks) / v2_clicks * 100,
        (df_plot[df_plot['variant_list']=='V1']['分组sku曝光总次数'].values[0] - v2_exposure) / v2_exposure * 100,
        (df_plot[df_plot['variant_list']=='V1']['分组加入购物车立即购买总次数'].values[0] - v2_cart) / v2_cart * 100,
        (df_plot[df_plot['variant_list']=='V1']['新SKU点击次数'].values[0] - v2_new_click) / v2_new_click * 100,
        df_plot[df_plot['variant_list']=='V1']['新SKU点击占比_vs_V2'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V1']['V2点击率对比'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V1']['V2加入购物车立即购买率对比'].values[0] * 100
    ]
    v3_changes = [
        (df_plot[df_plot['variant_list']=='V3']['分组sku点击总次数'].values[0] - v2_clicks) / v2_clicks * 100,
        (df_plot[df_plot['variant_list']=='V3']['分组sku曝光总次数'].values[0] - v2_exposure) / v2_exposure * 100,
        (df_plot[df_plot['variant_list']=='V3']['分组加入购物车立即购买总次数'].values[0] - v2_cart) / v2_cart * 100,
        (df_plot[df_plot['variant_list']=='V3']['新SKU点击次数'].values[0] - v2_new_click) / v2_new_click * 100,
        df_plot[df_plot['variant_list']=='V3']['新SKU点击占比_vs_V2'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V3']['V2点击率对比'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V3']['V2加入购物车立即购买率对比'].values[0] * 100
    ]
    v4_changes = [
        (df_plot[df_plot['variant_list']=='V4']['分组sku点击总次数'].values[0] - v2_clicks) / v2_clicks * 100,
        (df_plot[df_plot['variant_list']=='V4']['分组sku曝光总次数'].values[0] - v2_exposure) / v2_exposure * 100,
        (df_plot[df_plot['variant_list']=='V4']['分组加入购物车立即购买总次数'].values[0] - v2_cart) / v2_cart * 100,
        (df_plot[df_plot['variant_list']=='V4']['新SKU点击次数'].values[0] - v2_new_click) / v2_new_click * 100,
        df_plot[df_plot['variant_list']=='V4']['新SKU点击占比_vs_V2'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V4']['V2点击率对比'].values[0] * 100,
        df_plot[df_plot['variant_list']=='V4']['V2加入购物车立即购买率对比'].values[0] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.25
    bars_v1 = ax8.bar(x - width, v1_changes, width, label='V1 vs V2', color='#3498db', alpha=0.85)
    bars_v3 = ax8.bar(x, v3_changes, width, label='V3 vs V2', color='#27ae60', alpha=0.85)
    bars_v4 = ax8.bar(x + width, v4_changes, width, label='V4 vs V2', color='#9b59b6', alpha=0.85)
    ax8.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=2, label='V2基准线')
    ax8.set_ylabel('相对V2变化率 (%)', fontsize=12)
    ax8.set_title('各实验组 vs V2 综合变化率对比', fontsize=14, fontweight='bold', pad=15)
    ax8.set_xticks(x)
    ax8.set_xticklabels(metrics, rotation=30, ha='right', fontsize=10)
    ax8.legend(loc='upper right', fontsize=10)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.grid(axis='y', alpha=0.3)

    # ---------- 图9: 关键结论摘要 (动态数值版) ----------
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    # 动态提取数值辅助函数 (索引对应 metrics 列表)
    # metrics = ['点击次数', '曝光次数', '加购次数', '新SKU点击', '新SKU占比', '点击率', '转化率']
    def format_val(val):
        return f"{val:+.1f}%"

    summary_text = f"""
    【实验组 vs V2(对照组) 关键发现】

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    V1 (对比组):
       • 点击率: {format_val(v1_changes[5])}     转化率: {format_val(v1_changes[6])}
       • 新SKU占比: {format_val(v1_changes[4])}  (发现新品能力提升)

    V3 (隐式转换版 - Clickbased):
       • 点击率: {format_val(v3_changes[5])}     转化率: {format_val(v3_changes[6])}
       • 新SKU占比: {format_val(v3_changes[4])}  (最佳新品推荐)
       • 点击次数: {format_val(v3_changes[0])}

    V4 (隐式转换版 - Clickbased):
       • 点击率: {format_val(v4_changes[5])}     转化率: {format_val(v4_changes[6])}
       • 新SKU占比: {format_val(v4_changes[4])}  (新品曝光最多)
       • 点击次数: {format_val(v4_changes[0])}   (点击量最高)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    结论:
       • {"V3/V4" if v3_changes[4] > 0 and v4_changes[4] > 0 else "部分实验组"} 在新品发现能力上显著优于V2
       • V2 在转化效率上仍保持领先
       • 建议: 考虑 {"V3" if abs(v3_changes[6]) < abs(v4_changes[6]) else "V4"} 作为业务平衡方案
    """

    ax9.text(
        0.05, 0.95, summary_text,
        transform=ax9.transAxes,
        fontsize=13,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='#f8f9fa',
            edgecolor='#dee2e6',
            alpha=0.9
        )
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('ab_experiment_comparison_final.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    print("\n图表已保存: ab_experiment_comparison_final.png")