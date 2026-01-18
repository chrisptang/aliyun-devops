#!/usr/bin/env python
# coding: utf-8
"""
分类页LTR重排序AB实验 - 订单转化分析脚本

功能：
1. 基础订单转化分析
2. 按Category1维度的订单转化分析
3. 用户下单金额分布统计（P50, P75, P90, P99, P999, STD）
4. 变体间Category1增量分析
5. 用户下单金额分层分析

使用方法：
    python category-ltr-order-conversion-analytics.py
    python category-ltr-order-conversion-analytics.py --start-date 2026-01-07
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ============================================================================
# 命令行参数解析
# ============================================================================
parser = argparse.ArgumentParser(description='分类页LTR重排序AB实验 - 订单转化分析')
parser.add_argument('--start-date', type=str, default='20260107',
                    help='开始日期，格式YYYYMMDD (默认: 20260107)')
parser.add_argument('--output-dir', type=str, default='./data',
                    help='输出目录 (默认: ./data)')

args = parser.parse_args()
START_DATE = args.start_date.replace('-', '')
OUTPUT_DIR = args.output_dir

# ============================================================================
# 全局matplotlib字体配置 - 支持中文显示（Mac系统优化）
# ============================================================================
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

if sys.platform == 'darwin':
    # Mac系统：使用系统内置字体
    matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'STHeiti', 'Songti SC', 'Arial Unicode MS']
else:
    # 其他系统（Windows/Linux）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'

import matplotlib.pyplot as plt
import seaborn as sns

# 设置pandas显示选项
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)

# ============================================================================
# ODPS客户端导入
# ============================================================================
from odps_client import get_odps_sql_result_as_df

print("=" * 80)
print(f"  分类页LTR重排序AB实验 - 订单转化分析")
print(f"  开始日期: {START_DATE}")
print("=" * 80)


# ============================================================================
# 1. 基础订单转化分析
# ============================================================================
print("\n" + "=" * 80)
print("【1】基础订单转化分析")
print("=" * 80)

sql_basic_conversion = f"""
SELECT  variant_list
        ,COUNT(DISTINCT u.cust_id) AS total_uv
        ,COUNT(DISTINCT CASE WHEN o.order_no IS NOT NULL THEN o.cust_id END) AS order_uv
        ,COUNT(DISTINCT o.order_no) AS order_count
        ,COALESCE(SUM(real_total_amt), 0) AS total_gmv
        ,ROUND(COALESCE(SUM(real_total_amt), 0) / NULLIF(COUNT(DISTINCT o.order_no), 0), 2) AS avg_order_amt
        ,ROUND(COALESCE(SUM(real_total_amt), 0) / NULLIF(COUNT(DISTINCT u.cust_id), 0), 2) AS arpu
FROM    (
            SELECT  CAST(uid AS BIGINT) AS cust_id
                    ,COLLECT_SET(variant_list) variant_list
            FROM    summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df
            WHERE   ds = MAX_PT('summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df')
            GROUP BY uid
        ) u
LEFT JOIN summerfarm_tech.dwd_trd_order_df o
ON      o.ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
AND     o.cust_id = u.cust_id
AND     o.order_date >= '{START_DATE}'
AND     o.order_status IN (2, 3, 6)
GROUP BY variant_list
ORDER BY variant_list
"""

print("执行SQL查询...")
df_basic = get_odps_sql_result_as_df(sql_basic_conversion)
df_basic.columns = ['变体', '总UV', '下单UV', '订单总数', '下单总金额', '平均订单金额', 'ARPU']

# 计算转化率
df_basic['转化率(%)'] = round(100.0 * df_basic['下单UV'] / df_basic['总UV'], 2)

print("\n基础订单转化数据:")
print(df_basic.to_string(index=False))


# ============================================================================
# 2. 按Category1维度的订单转化分析
# ============================================================================
print("\n" + "=" * 80)
print("【2】按Category1维度的订单转化分析")
print("=" * 80)

sql_category_conversion = f"""
SELECT  variant_list
        ,category1
        ,COUNT(DISTINCT o.cust_id) AS order_uv
        ,COUNT(DISTINCT o.order_no) AS order_count
        ,SUM(real_total_amt) AS total_gmv
        ,ROUND(SUM(real_total_amt) / NULLIF(COUNT(DISTINCT o.order_no), 0), 2) AS avg_order_amt
        ,SUM(sku_cnt) AS sku_count
FROM    (
            SELECT  CAST(uid AS BIGINT) AS cust_id
                    ,COLLECT_SET(variant_list) variant_list
            FROM    summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df
            WHERE   ds = MAX_PT('summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df')
            GROUP BY uid
        ) u
INNER JOIN summerfarm_tech.dwd_trd_order_df o
ON      o.ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
AND     o.cust_id = u.cust_id
AND     o.order_date >= '{START_DATE}'
AND     o.order_status IN (2, 3, 6)
WHERE   o.category1 IS NOT NULL
GROUP BY variant_list, category1
ORDER BY variant_list, total_gmv DESC
"""

print("执行SQL查询...")
df_category = get_odps_sql_result_as_df(sql_category_conversion)
df_category.columns = ['变体', '一级分类', '下单UV', '订单总数', '下单总金额', '平均订单金额', 'SKU数量']

print("\n按Category1的订单转化数据（前20行）:")
print(df_category.head(20).to_string(index=False))


# ============================================================================
# 3. 用户下单金额分布统计
# ============================================================================
print("\n" + "=" * 80)
print("【3】用户下单金额分布统计（P50, P75, P90, P99, P999, STD）")
print("=" * 80)

sql_amount_distribution = f"""
SELECT  variant_list
        ,COUNT(DISTINCT cust_id) AS user_count
        ,ROUND(AVG(user_total_amt), 2) AS avg_amt
        ,ROUND(STDDEV(user_total_amt), 2) AS std_amt
        ,ROUND(PERCENTILE(CAST(user_total_amt * 100 AS BIGINT), 0.50) / 100.0, 2) AS p50
        ,ROUND(PERCENTILE(CAST(user_total_amt * 100 AS BIGINT), 0.75) / 100.0, 2) AS p75
        ,ROUND(PERCENTILE(CAST(user_total_amt * 100 AS BIGINT), 0.90) / 100.0, 2) AS p90
        ,ROUND(PERCENTILE(CAST(user_total_amt * 100 AS BIGINT), 0.99) / 100.0, 2) AS p99
        ,ROUND(PERCENTILE(CAST(user_total_amt * 100 AS BIGINT), 0.999) / 100.0, 2) AS p999
        ,ROUND(MIN(user_total_amt), 2) AS min_amt
        ,ROUND(MAX(user_total_amt), 2) AS max_amt
FROM    (
            SELECT  u.variant_list
                    ,o.cust_id
                    ,SUM(real_total_amt) AS user_total_amt
            FROM    (
                        SELECT  CAST(uid AS BIGINT) AS cust_id
                                ,COLLECT_SET(variant_list) variant_list
                        FROM    summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df
                        WHERE   ds = MAX_PT('summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df')
                        GROUP BY uid
                    ) u
            INNER JOIN summerfarm_tech.dwd_trd_order_df o
            ON      o.ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
            AND     o.cust_id = u.cust_id
            AND     o.order_date >= '{START_DATE}'
            AND     o.order_status IN (2, 3, 6)
            GROUP BY u.variant_list, o.cust_id
        ) user_order_summary
GROUP BY variant_list
ORDER BY variant_list
"""

print("执行SQL查询...")
df_distribution = get_odps_sql_result_as_df(sql_amount_distribution)
df_distribution.columns = ['变体', '下单用户数', 'AVG', 'STD', 'P50', 'P75', 'P90', 'P99', 'P999', 'MIN', 'MAX']

print("\n用户下单金额分布统计:")
print(df_distribution.to_string(index=False))


# ============================================================================
# 4. 变体间Category1增量分析（相对于V2对照组）
# ============================================================================
print("\n" + "=" * 80)
print("【4】变体间Category1增量分析（相对于V2对照组）")
print("=" * 80)

sql_category_lift = f"""
WITH category_stats AS (
    SELECT  CONCAT('[', u.variant_list[0], ']') AS variant_list
            ,o.category1
            ,COUNT(DISTINCT o.cust_id) AS order_uv
            ,COUNT(DISTINCT o.order_no) AS order_count
            ,SUM(o.real_total_amt) AS total_gmv
    FROM    (
                SELECT  CAST(uid AS BIGINT) AS cust_id
                        ,COLLECT_SET(variant_list) variant_list
                FROM    summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df
                WHERE   ds = MAX_PT('summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df')
                GROUP BY uid
            ) u
    INNER JOIN summerfarm_tech.dwd_trd_order_df o
    ON      o.ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
    AND     o.cust_id = u.cust_id
    AND     o.order_date >= '{START_DATE}'
    AND     o.order_status IN (2, 3, 6)
    WHERE   o.category1 IS NOT NULL
    GROUP BY CONCAT('[', u.variant_list[0], ']'), o.category1
),
v2_baseline AS (
    SELECT  category1
            ,order_uv AS v2_uv
            ,order_count AS v2_orders
            ,total_gmv AS v2_gmv
    FROM    category_stats
    WHERE   variant_list = '[V2]'
)
SELECT  cs.variant_list
        ,cs.category1
        ,cs.order_uv
        ,v2.v2_uv
        ,ROUND(100.0 * (cs.order_uv - v2.v2_uv) / NULLIF(v2.v2_uv, 0), 2) AS uv_lift_pct
        ,cs.total_gmv
        ,v2.v2_gmv
        ,ROUND(100.0 * (cs.total_gmv - v2.v2_gmv) / NULLIF(v2.v2_gmv, 0), 2) AS gmv_lift_pct
FROM    category_stats cs
LEFT JOIN v2_baseline v2
ON      cs.category1 = v2.category1
WHERE   cs.variant_list != '[V2]'
ORDER BY cs.variant_list, cs.total_gmv DESC
"""

print("执行SQL查询...")
df_lift = get_odps_sql_result_as_df(sql_category_lift)
df_lift.columns = ['变体', '一级分类', '下单UV', 'V2下单UV', 'UV提升(%)', '下单金额', 'V2下单金额', 'GMV提升(%)']

print("\n变体间Category1增量分析（前30行）:")
print(df_lift.head(30).to_string(index=False))


# ============================================================================
# 5. 用户下单金额分层分析
# ============================================================================
print("\n" + "=" * 80)
print("【5】用户下单金额分层分析")
print("=" * 80)

sql_amount_tier = f"""
SELECT  variant_list
        ,amt_tier
        ,user_count
        ,ROUND(100.0 * user_count / SUM(user_count) OVER (PARTITION BY variant_list), 2) AS pct
        ,tier_gmv
        ,ROUND(tier_gmv / user_count, 2) AS tier_avg_amt
FROM    (
            SELECT  variant_list
                    ,CASE
                        WHEN user_total_amt < 100 THEN '1. <100'
                        WHEN user_total_amt < 500 THEN '2. 100-500'
                        WHEN user_total_amt < 1000 THEN '3. 500-1000'
                        WHEN user_total_amt < 2000 THEN '4. 1000-2000'
                        WHEN user_total_amt < 5000 THEN '5. 2000-5000'
                        ELSE '6. >=5000'
                    END AS amt_tier
                    ,COUNT(DISTINCT cust_id) AS user_count
                    ,SUM(user_total_amt) AS tier_gmv
            FROM    (
                        SELECT  u.variant_list
                                ,o.cust_id
                                ,SUM(real_total_amt) AS user_total_amt
                        FROM    (
                                    SELECT  CAST(uid AS BIGINT) AS cust_id
                                            ,COLLECT_SET(variant_list) variant_list
                                    FROM    summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df
                                    WHERE   ds = MAX_PT('summerfarm_ds.temp_category_ltr_ab_all_user_view_data_df')
                                    GROUP BY uid
                                ) u
                        INNER JOIN summerfarm_tech.dwd_trd_order_df o
                        ON      o.ds = MAX_PT('summerfarm_tech.dwd_trd_order_df')
                        AND     o.cust_id = u.cust_id
                        AND     o.order_date >= '{START_DATE}'
                        AND     o.order_status IN (2, 3, 6)
                        GROUP BY u.variant_list, o.cust_id
                    ) user_amt
            GROUP BY variant_list,
                    CASE
                        WHEN user_total_amt < 100 THEN '1. <100'
                        WHEN user_total_amt < 500 THEN '2. 100-500'
                        WHEN user_total_amt < 1000 THEN '3. 500-1000'
                        WHEN user_total_amt < 2000 THEN '4. 1000-2000'
                        WHEN user_total_amt < 5000 THEN '5. 2000-5000'
                        ELSE '6. >=5000'
                    END
        ) t
ORDER BY variant_list, amt_tier
"""

print("执行SQL查询...")
df_tier = get_odps_sql_result_as_df(sql_amount_tier)
df_tier.columns = ['变体', '金额区间', '用户数', '占比(%)', '区间GMV', '区间人均金额']

print("\n用户下单金额分层分析:")
print(df_tier.to_string(index=False))


# ============================================================================
# 6. 可视化分析
# ============================================================================
print("\n" + "=" * 80)
print("【6】生成可视化图表")
print("=" * 80)


def setup_chinese_font():
    """重新设置中文字体（在seaborn设置之后调用）"""
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang HK', 'STHeiti', 'Songti SC', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.family'] = 'sans-serif'


def create_conversion_visualization(df_basic, df_distribution, df_tier, df_category):
    """
    创建订单转化分析的综合可视化图表
    """
    # 设置seaborn主题
    sns.set_theme(style="whitegrid", palette="husl")
    setup_chinese_font()

    # 颜色方案
    colors = {'V1': '#3498db', 'V2': '#e74c3c', 'V3': '#27ae60', 'V4': '#9b59b6'}

    # 将所有数值列转换为float，避免Decimal类型问题
    for col in df_basic.select_dtypes(include=['object']).columns:
        if col != '变体':
            try:
                df_basic[col] = pd.to_numeric(df_basic[col], errors='ignore')
            except:
                pass

    for col in df_distribution.columns:
        if col != '变体':
            try:
                df_distribution[col] = pd.to_numeric(df_distribution[col], errors='coerce').astype(float)
            except:
                pass

    for col in df_tier.columns:
        if col not in ['变体', '金额区间']:
            try:
                df_tier[col] = pd.to_numeric(df_tier[col], errors='coerce').astype(float)
            except:
                pass

    for col in df_category.columns:
        if col not in ['变体', '一级分类']:
            try:
                df_category[col] = pd.to_numeric(df_category[col], errors='coerce').astype(float)
            except:
                pass

    # 提取变体名称（简化显示）
    def simplify_variant(v):
        if 'V1' in str(v):
            return 'V1'
        elif 'V2' in str(v):
            return 'V2'
        elif 'V3' in str(v):
            return 'V3'
        elif 'V4' in str(v):
            return 'V4'
        return str(v)

    # ========================================================================
    # 图1: 基础转化指标对比
    # ========================================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('分类页LTR重排序AB实验 - 基础订单转化指标对比', fontsize=16, fontweight='bold', y=1.02)

    df_basic_plot = df_basic.copy()
    df_basic_plot['变体简称'] = df_basic_plot['变体'].apply(simplify_variant)
    variant_colors = [colors.get(v, '#95a5a6') for v in df_basic_plot['变体简称']]

    # 1.1 总UV vs 下单UV
    ax = axes1[0, 0]
    x = np.arange(len(df_basic_plot))
    width = 0.35
    bars1 = ax.bar(x - width/2, df_basic_plot['总UV'], width, label='总UV', color='#bdc3c7', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_basic_plot['下单UV'], width, label='下单UV', color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('用户数')
    ax.set_title('总UV vs 下单UV', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_basic_plot['变体简称'])
    ax.legend()
    for bar, val in zip(bars1, df_basic_plot['总UV']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(val):,}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, df_basic_plot['下单UV']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(val):,}',
                ha='center', va='bottom', fontsize=9)

    # 1.2 转化率
    ax = axes1[0, 1]
    bars = ax.bar(df_basic_plot['变体简称'], df_basic_plot['转化率(%)'], color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('转化率 (%)')
    ax.set_title('下单转化率', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df_basic_plot['转化率(%)']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.3 下单总金额
    ax = axes1[0, 2]
    bars = ax.bar(df_basic_plot['变体简称'], df_basic_plot['下单总金额'], color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('下单总金额 (GMV)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df_basic_plot['下单总金额']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:,.0f}',
                ha='center', va='bottom', fontsize=9)

    # 1.4 平均订单金额
    ax = axes1[1, 0]
    bars = ax.bar(df_basic_plot['变体简称'], df_basic_plot['平均订单金额'], color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('平均订单金额', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df_basic_plot['平均订单金额']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.5 ARPU
    ax = axes1[1, 1]
    bars = ax.bar(df_basic_plot['变体简称'], df_basic_plot['ARPU'], color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('ARPU (人均收入)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df_basic_plot['ARPU']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.6 订单总数
    ax = axes1[1, 2]
    bars = ax.bar(df_basic_plot['变体简称'], df_basic_plot['订单总数'], color=variant_colors, edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('订单数')
    ax.set_title('订单总数', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, df_basic_plot['订单总数']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(val):,}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/order_conversion_basic.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"基础转化指标图表已保存: {OUTPUT_DIR}/order_conversion_basic.png")
    plt.close(fig1)

    # ========================================================================
    # 图2: 用户下单金额分布统计
    # ========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('分类页LTR重排序AB实验 - 用户下单金额分布统计', fontsize=16, fontweight='bold', y=1.02)

    df_dist_plot = df_distribution.copy()
    df_dist_plot['变体简称'] = df_dist_plot['变体'].apply(simplify_variant)
    variant_colors_dist = [colors.get(v, '#95a5a6') for v in df_dist_plot['变体简称']]

    # 2.1 均值和标准差
    ax = axes2[0, 0]
    x = np.arange(len(df_dist_plot))
    bars = ax.bar(x, df_dist_plot['AVG'], yerr=df_dist_plot['STD'], capsize=5,
                  color=variant_colors_dist, edgecolor='black', alpha=0.8)
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('人均下单金额 (AVG ± STD)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_dist_plot['变体简称'])
    for bar, avg, std in zip(bars, df_dist_plot['AVG'], df_dist_plot['STD']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 10,
                f'{avg:.0f}\n±{std:.0f}', ha='center', va='bottom', fontsize=9)

    # 2.2 分位数对比 (P50, P75, P90)
    ax = axes2[0, 1]
    x = np.arange(len(df_dist_plot))
    width = 0.25
    bars1 = ax.bar(x - width, df_dist_plot['P50'], width, label='P50', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, df_dist_plot['P75'], width, label='P75', color='#27ae60', edgecolor='black')
    bars3 = ax.bar(x + width, df_dist_plot['P90'], width, label='P90', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('下单金额分位数对比 (P50/P75/P90)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_dist_plot['变体简称'])
    ax.legend()

    # 2.3 高分位数对比 (P99, P999)
    ax = axes2[1, 0]
    x = np.arange(len(df_dist_plot))
    width = 0.35
    bars1 = ax.bar(x - width/2, df_dist_plot['P99'], width, label='P99', color='#9b59b6', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_dist_plot['P999'], width, label='P999', color='#e67e22', edgecolor='black')
    ax.set_xlabel('实验分组')
    ax.set_ylabel('金额 (元)')
    ax.set_title('高消费用户分位数对比 (P99/P999)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_dist_plot['变体简称'])
    ax.legend()
    for bar, val in zip(bars1, df_dist_plot['P99']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:,.0f}',
                ha='center', va='bottom', fontsize=9, rotation=45)
    for bar, val in zip(bars2, df_dist_plot['P999']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:,.0f}',
                ha='center', va='bottom', fontsize=9, rotation=45)

    # 2.4 分布统计汇总表
    ax = axes2[1, 1]
    ax.axis('off')
    table_data = df_dist_plot[['变体简称', 'AVG', 'STD', 'P50', 'P75', 'P90', 'P99', 'P999']].values
    table_headers = ['变体', 'AVG', 'STD', 'P50', 'P75', 'P90', 'P99', 'P999']
    table = ax.table(cellText=table_data, colLabels=table_headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('下单金额分布统计汇总', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/order_amount_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"金额分布统计图表已保存: {OUTPUT_DIR}/order_amount_distribution.png")
    plt.close(fig2)

    # ========================================================================
    # 图3: 金额分层分析
    # ========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle('分类页LTR重排序AB实验 - 用户下单金额分层分析', fontsize=16, fontweight='bold', y=1.02)

    df_tier_plot = df_tier.copy()
    df_tier_plot['变体简称'] = df_tier_plot['变体'].apply(simplify_variant)

    # 3.1 各变体金额分层用户占比
    ax = axes3[0]
    tier_pivot = df_tier_plot.pivot(index='金额区间', columns='变体简称', values='占比(%)')
    tier_pivot = tier_pivot.reindex(['1. <100', '2. 100-500', '3. 500-1000', '4. 1000-2000', '5. 2000-5000', '6. >=5000'])
    tier_pivot.plot(kind='bar', ax=ax, color=[colors.get(c, '#95a5a6') for c in tier_pivot.columns],
                    edgecolor='black', width=0.8)
    ax.set_xlabel('金额区间')
    ax.set_ylabel('用户占比 (%)')
    ax.set_title('各变体金额分层用户占比', fontsize=12, fontweight='bold')
    ax.legend(title='变体')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # 3.2 各变体金额分层GMV
    ax = axes3[1]
    gmv_pivot = df_tier_plot.pivot(index='金额区间', columns='变体简称', values='区间GMV')
    gmv_pivot = gmv_pivot.reindex(['1. <100', '2. 100-500', '3. 500-1000', '4. 1000-2000', '5. 2000-5000', '6. >=5000'])
    gmv_pivot.plot(kind='bar', ax=ax, color=[colors.get(c, '#95a5a6') for c in gmv_pivot.columns],
                   edgecolor='black', width=0.8)
    ax.set_xlabel('金额区间')
    ax.set_ylabel('GMV (元)')
    ax.set_title('各变体金额分层GMV', fontsize=12, fontweight='bold')
    ax.legend(title='变体')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/order_amount_tier.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"金额分层分析图表已保存: {OUTPUT_DIR}/order_amount_tier.png")
    plt.close(fig3)

    # ========================================================================
    # 图4: Top Category1 分析
    # ========================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(18, 14))
    fig4.suptitle('分类页LTR重排序AB实验 - 各变体Top10品类分析', fontsize=16, fontweight='bold', y=1.02)

    df_cat_plot = df_category.copy()
    df_cat_plot['变体简称'] = df_cat_plot['变体'].apply(simplify_variant)

    variants_list = ['V1', 'V2', 'V3', 'V4']
    for idx, variant in enumerate(variants_list):
        ax = axes4[idx // 2, idx % 2]
        variant_data = df_cat_plot[df_cat_plot['变体简称'] == variant].nlargest(10, '下单总金额')

        if not variant_data.empty:
            bars = ax.barh(variant_data['一级分类'], variant_data['下单总金额'],
                          color=colors.get(variant, '#95a5a6'), edgecolor='black', alpha=0.8)
            ax.set_xlabel('下单总金额 (元)')
            ax.set_title(f'{variant} - Top10品类GMV', fontsize=12, fontweight='bold')
            ax.invert_yaxis()

            for bar, val in zip(bars, variant_data['下单总金额']):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2., f'{val:,.0f}',
                        ha='left', va='center', fontsize=9)

    plt.tight_layout()
    fig4.savefig(f'{OUTPUT_DIR}/order_category_top10.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Top10品类分析图表已保存: {OUTPUT_DIR}/order_category_top10.png")
    plt.close(fig4)

    print("\n所有可视化图表生成完成！")


# 执行可视化
create_conversion_visualization(df_basic, df_distribution, df_tier, df_category)


# ============================================================================
# 7. 保存结果到CSV
# ============================================================================
print("\n" + "=" * 80)
print("【7】保存分析结果到CSV")
print("=" * 80)

df_basic.to_csv(f'{OUTPUT_DIR}/order_conversion_basic.csv', index=False, encoding='utf-8-sig')
df_category.to_csv(f'{OUTPUT_DIR}/order_conversion_category.csv', index=False, encoding='utf-8-sig')
df_distribution.to_csv(f'{OUTPUT_DIR}/order_amount_distribution.csv', index=False, encoding='utf-8-sig')
df_tier.to_csv(f'{OUTPUT_DIR}/order_amount_tier.csv', index=False, encoding='utf-8-sig')
df_lift.to_csv(f'{OUTPUT_DIR}/order_category_lift.csv', index=False, encoding='utf-8-sig')

print(f"基础转化数据已保存: {OUTPUT_DIR}/order_conversion_basic.csv")
print(f"品类转化数据已保存: {OUTPUT_DIR}/order_conversion_category.csv")
print(f"金额分布数据已保存: {OUTPUT_DIR}/order_amount_distribution.csv")
print(f"金额分层数据已保存: {OUTPUT_DIR}/order_amount_tier.csv")
print(f"品类增量数据已保存: {OUTPUT_DIR}/order_category_lift.csv")


# ============================================================================
# 8. 打印分析结论
# ============================================================================
print("\n" + "=" * 80)
print("【8】分析结论")
print("=" * 80)

# 找出转化率最高的变体
best_conversion = df_basic.loc[df_basic['转化率(%)'].idxmax()]
print(f"\n转化率最高的变体: {best_conversion['变体']} ({best_conversion['转化率(%)']:.2f}%)")

# 找出ARPU最高的变体
best_arpu = df_basic.loc[df_basic['ARPU'].idxmax()]
print(f"ARPU最高的变体: {best_arpu['变体']} ({best_arpu['ARPU']:.2f}元)")

# 找出GMV最高的变体
best_gmv = df_basic.loc[df_basic['下单总金额'].idxmax()]
print(f"GMV最高的变体: {best_gmv['变体']} ({best_gmv['下单总金额']:,.2f}元)")

# V2对照组数据
v2_mask = df_basic['变体'].astype(str).str.contains('V2')
v2_data = df_basic[v2_mask]
if not v2_data.empty:
    v2_conversion = float(v2_data['转化率(%)'].values[0])
    v2_arpu = float(v2_data['ARPU'].values[0])
    v2_gmv = float(v2_data['下单总金额'].values[0])

    print(f"\n相对于V2对照组的提升:")
    for _, row in df_basic.iterrows():
        variant_str = str(row['变体'])
        if 'V2' not in variant_str:
            conv_lift = (float(row['转化率(%)']) - v2_conversion) / v2_conversion * 100 if v2_conversion else 0
            arpu_lift = (float(row['ARPU']) - v2_arpu) / v2_arpu * 100 if v2_arpu else 0
            gmv_lift = (float(row['下单总金额']) - v2_gmv) / v2_gmv * 100 if v2_gmv else 0
            print(f"  {variant_str}: 转化率 {conv_lift:+.2f}%, ARPU {arpu_lift:+.2f}%, GMV {gmv_lift:+.2f}%")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
