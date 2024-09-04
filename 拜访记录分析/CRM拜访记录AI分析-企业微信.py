#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os

# Expand the `~` to the full path and append it to `sys.path`
full_path = os.path.expanduser("~/Documents/github/aliyun-devops")
sys.path.append(full_path)

from odps_client import get_odps_sql_result_as_df
from datetime import datetime, timedelta

import pandas as pd

ds = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

# Create a directory for the markdown files if it doesn't exist
output_dir = f"output_wecomm_{ds}"
os.makedirs(output_dir, exist_ok=True)


# In[ ]:


# Import the base64 encoding library.
import base64, os, time
import logging

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

proxy_object = {"http": "http://127.0.0.1:8001", "https": "http://127.0.0.1:8001"}


from openai import AzureOpenAI

client_gpt4o = AzureOpenAI(
    api_version="2024-03-01-preview",
    azure_endpoint="https://xm-ai-us2.openai.azure.com",
    api_key=os.getenv("AZURE_GPT4O_API_KEY", ""),
)

client_gpt4o_mini = AzureOpenAI(
    api_version="2024-03-01-preview",
    azure_endpoint="https://xm-ai-us.openai.azure.com",
    api_key=os.getenv("AZURE_GPT4O_MINI_API_KEY", ""),
)


def call_azure_openai(
    messages=[], retrying=1, is_gpt4o=False, json=True, max_tokens=16384
) -> (str, bool):
    if retrying < 0:
        return "超过了最大重试次数", False
    completion = None
    ## gpt3.5:  gpt-35-turbo-16k,
    ## got4o:   gpt-4o
    ## got4o-mini:   gpt-4o-mini
    model = "gpt-4o-mini"
    client_to_use = client_gpt4o_mini
    if is_gpt4o:
        logging.info(f"using GPT-4o...:{messages}")
        model = "gpt-4o"
        client_to_use = client_gpt4o
    try:
        completion = client_to_use.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
            messages=messages,
            response_format={"type": "json_object"} if json else {"type": "text"},
        )
        response = completion.choices[0].message.content
        if (
            len(completion.choices) <= 0
            or f"{completion.choices[0].finish_reason}" == "content_filter"
        ):
            return f"azure过滤了本次请求:{completion.choices[0].to_dict()}", False
        if response is None:
            logging.info(f"azure API返回了异常:{completion.to_dict()}")
            time.sleep(10)
            return call_azure_openai(
                messages=messages,
                retrying=retrying - 1,
                is_gpt4o=is_gpt4o,
            )
        logging.info(f"total usage:{completion.usage}")
        return response, True
    except Exception as e:
        logging.info(
            f"请求azure接口报错了:{e}\n messages:{messages}, completion:{completion}"
        )
        if retrying <= 0 or "Error code: 400" in f"{e}":
            return f"{e}", False
        logging.info(f"重试中...{retrying}, messages:{messages}")
        return call_azure_openai(
            messages=messages,
            retrying=retrying - 1,
            is_gpt4o=is_gpt4o,
        )


def call_ai_api_to_get_extract_visit_info(visit_text=""):
    result = {}
    json_text, is_ok = call_azure_openai(
        is_gpt4o=False,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": """
用户会给发给你一系列销售员对客户的拜访记录，请你用JSON回答以下几个问题：
- 客户是否有跟销售员互动？
- 销售向客户推荐了哪些具体的活动？
- 销售向客户推荐了哪些具体的商品？
- 客户的主要采买渠道？
- 客户对公司的产品有什么看法？(请列明具体的产品名)
- 客户对公司的配送服务有什么看法？
- 客户对公司的评价是怎样的？（正向/负向/中立/无法判断）
- 客户不愿意下单的原因？
- 销售员本次拜访的主要目的？
- 销售员解决了客户哪些问题？
- 拜访记录完整性打分？（0-100分，100分表示非常完整，0分表示非常不完整）

**请注意，‘安佳’，‘铁塔’一般来说是商品名字，而不太可能是活动名字，活动名字一般带有‘专享’、‘清仓’、‘特价’、‘活动’等字样**
**请你完全基于销售员的拜访记录内容来回答以上问题，如果拜访内容中找不到问题的答案，请回答‘无’**
**请你用问题的标题做JSON的key，答案做value，比如：**
{
  "客户是否有跟销售员互动": "是,交流了2句话",
  "销售向客户推荐了哪些具体的活动": "无",
  "销售向客户推荐了哪些具体的商品": "无",
  "客户的主要采买渠道": "无",
  "客户对公司的产品有什么看法": "无",
  "客户对公司的配送服务有什么看法 "无",
  "客户对公司的评价是怎样的": "无法判断",
  "客户不愿意下单的原因": "无",
  "销售员本次拜访的主要目的": "通知客户服务升级",
  "销售员解决了客户哪些问题": "无",
  "拜访记录完整性打分": 10,
}
""",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": visit_text}],
            },
        ],
    )

    logging.info(f"json_text:{json_text}, visit_text:{visit_text}")
    return json_text


from datetime import datetime

date_of_now = datetime.now().strftime("%Y-%m-%d")


# ## 先获取聊天记录
# 

# In[ ]:


import json

sql=f"""
SELECT  ds
        ,form AS msg_from
        ,tolist
        ,body
        ,msgtime
FROM    summerfarm_tech.dwd_log_wecom_di
WHERE   ds = '{ds}'
;
"""

wecom_log_df = get_odps_sql_result_as_df(sql=sql)
wecom_log_df['tolist'] = wecom_log_df['tolist'].apply(lambda x: json.loads(x if x is not None else '[""]')[0])
# Display the top 20 rows of the wecom_log_df dataframe
wecom_log_df['conversation_group'] = wecom_log_df.apply(lambda row: '-'.join(sorted([f"{row['msg_from']}", f"{row['tolist']}"])), axis=1)
# wecom_log_df.head(20)


# In[ ]:


# --------------------绑定企微的客户名单
sql_wecom=f"""
SELECT  oc.m_id
        ,c.cust_name
        ,array_join(collect_set(wx.user_id),',') 绑定微信ID
        ,wx.external_userid
        ,MAX(create_time) 绑定微信时间
FROM    summerfarm_tech.ods_merchant_sub_account_df oc
LEFT JOIN summerfarm_tech.dim_cust_df c
ON      oc.m_id = c.cust_id
AND     c.ds = MAX_PT('summerfarm_tech.dim_cust_df')
AND     c.ds BETWEEN c.start_at AND c.end_at
LEFT JOIN summerfarm_tech.ods_wechat_user_info_df wx
ON      oc.unionid = wx.unionid
AND     wx.ds = MAX_PT('summerfarm_tech.ods_wechat_user_info_df')
AND     wx.status = 1
WHERE   oc.unionid IS NOT NULL
AND     oc.ds = MAX_PT('summerfarm_tech.ods_merchant_sub_account_df')
AND     oc.type = 0
AND     wx.user_id IS NOT NULL
GROUP BY oc.m_id
         ,c.cust_name
         ,wx.external_userid
;
"""

wecom_user_df = get_odps_sql_result_as_df(sql=sql_wecom)
# wecom_user_df.head(20)


# In[ ]:


# Display all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

external_userid_to_mid_map = {}

for index, row in wecom_user_df.iterrows():
    external_userid_to_mid_map[row["external_userid"]] = (
        f"{row['cust_name']}:{row['m_id']}"
    )

conversation_groups = wecom_log_df["conversation_group"].unique()

all_conversation_list = []

for conversation_group in conversation_groups:
    # print(f"conversation_group:{conversation_group}")
    bd_wecom_id = None
    from_to_pair = conversation_group.split("-")
    if "wmndqQCQAA" not in from_to_pair[0]:
        bd_wecom_id = from_to_pair[0]
    elif len(from_to_pair) > 1 and "wmndqQCQAA" not in from_to_pair[1]:
        bd_wecom_id = from_to_pair[1]
    else:
        print(f"找不到BD ID:{conversation_group}")
    mid = None
    mname = None
    conversation_start_time = None
    conversation_end_time = None
    conversation_group_df = wecom_log_df[
        wecom_log_df["conversation_group"] == conversation_group
    ]

    conversation_group_df["msgtime"] = pd.to_datetime(conversation_group_df["msgtime"])
    conversation_group_df.sort_values(by="msgtime", ascending=True, inplace=True)
    conversation = {"conversation_group": conversation_group}
    conversation_list = []
    for index, row in conversation_group_df.iterrows():
        if conversation_start_time is None:
            conversation_start_time = row["msgtime"]
        conversation_end_time = row["msgtime"]
        msg_from = external_userid_to_mid_map.get(row["msg_from"], row["msg_from"])
        msg_to = external_userid_to_mid_map.get(row["tolist"], row["tolist"])
        if mid is None:
            if row["tolist"] is not None and "wmndqQCQAA" in row["tolist"]:
                mid = f"{msg_to}:{msg_to}".split(":")[1]
                mname = msg_to.split(":")[0]
            elif row["msg_from"] is not None and "wmndqQCQAA" in row["msg_from"]:
                mid = f"{msg_from}:{msg_from}".split(":")[1]
                mname = msg_from.split(":")[0]
        conversation_list.append(
            f"[{row['msgtime']}] {msg_from} -> {msg_to}: {json.loads(row['body'] or '{}').get('content','非文本')}"
        )
    conversation["conversation_list"] = conversation_list
    conversation["商户ID"] = mid
    conversation["商户名"] = mname
    conversation["会话开始时间"] = conversation_start_time
    conversation["会话结束时间"] = conversation_end_time
    conversation["BD企业微信ID"] = bd_wecom_id
    all_conversation_list.append(conversation)

all_conversation_df = pd.DataFrame(all_conversation_list)
all_conversation_df.to_csv(f"{output_dir}/all_conversation_df.csv", index=False)
# all_conversation_df.head(20)


# In[ ]:


staffname_list = ["白津源", "宋懿航", "李梦婷", "陈汉文"]
staffname_list = [f"'{staffname}'" for staffname in staffname_list]
staffname_list = ",".join(staffname_list)

mid_to_bd_sql = f"""
SELECT  b.cust_id
        ,b.cust_name 商户名
        ,c.bd_name
        ,c.bd_id
        ,b.abandon_date
        ,CASE   WHEN d.last_order_time IS NULL THEN '从未下单'
                ELSE '已下单'
        END AS 是否下过单
        ,DATEDIFF(GETDATE(),d.last_order_time,'dd') AS 距离上次下单天数
        ,od.历史下单数
        ,od.历史总下单金额
        ,DATEDIFF(GETDATE(),d.register_time,'dd') AS 注册天数
        ,c.m1_name AS M1负责人
        ,c.m2_name AS M2负责人
        ,c.m3_name AS M3负责人
        ,c.zone_name AS 销售区域
FROM    summerfarm_tech.dim_cust_df b
LEFT JOIN summerfarm_tech.dim_bd_df c
ON      c.bd_id = b.bd_id
AND     c.ds = MAX_PT('summerfarm_tech.dim_bd_df')
LEFT JOIN summerfarm_tech.ods_merchant_df d
ON      d.m_id = b.cust_id
AND     d.ds = MAX_PT('summerfarm_tech.ods_merchant_df')
LEFT JOIN   (
                SELECT  m_id
                        ,SUM(total_price) 历史总下单金额
                        ,COUNT(DISTINCT CASE    WHEN od.status IN (2,3,6) THEN od.order_no END) AS 历史下单数
                FROM    summerfarm_tech.ods_orders_df od
                WHERE   ds = MAX_PT('summerfarm_tech.ods_orders_df')
                GROUP BY m_id
            ) od
ON      od.m_id = b.cust_id
WHERE   b.ds = MAX_PT('summerfarm_tech.dim_cust_df')
AND     c.m1_name IS NOT NULL
AND     b.abandon_date = 99991231
AND     c.is_disabled = 0
;
"""
print(f"sql:{mid_to_bd_sql}")
mid_to_bd_df = get_odps_sql_result_as_df(sql=mid_to_bd_sql)
mid_to_bd_df.drop_duplicates(
    subset=["cust_id"], inplace=True
)

# mid_to_bd_df.head(20)


# In[ ]:


# Assuming all_conversation_df is already defined in the notebook
all_conversation_df['商户ID'] = all_conversation_df['商户ID'].astype(str)
mid_to_bd_df['cust_id'] = mid_to_bd_df['cust_id'].astype(str)

# Merge the dataframes and append '_y' to duplicate columns from the right dataframe
merged_df = all_conversation_df.merge(mid_to_bd_df, how='left', left_on='商户ID', right_on='cust_id', suffixes=('', '_y'))
merged_df = merged_df.dropna(subset=['cust_id'])
# merged_df.head(10)


# In[ ]:


system_prompt = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "你是一个资深销售主管，擅长分析销售员的客户拜访记录",
        }
    ],
}


def call_ai_api_to_get_insigns(city, csv_string=""):
    merged_markdown_result = ""
    text, is_ok = call_azure_openai(
        is_gpt4o=False,
        json=False,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""以下是你管理的团队销售员的客户拜访记录，作为销售主管，从中你发现了哪些值得注意的现象？
    将你发现的每一种现象按照重要程度倒序排列。请你列举数据以阐述其值得你关注的原因。
    **请你完全基于CSV的数据做分析，如果用户没有具体的反馈内容，请不要推测。这对公司来说非常重要，我们需要使用真实的客户反馈去调整经营策略**
    以下是CSV内容：\n\n{csv_string}""",
                    }
                ],
            },
        ],
    )

    if not is_ok:
        logging.info(f"call_ai_api_to_get_insigns failed: {text}")
        return ""

    merged_markdown_result = f"## {city}团队销售拜访记录AI分析\n\n{text}\n\n"

    text, is_ok = call_azure_openai(
        is_gpt4o=False,
        json=False,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""作为销售主管，请你分析在这些拜访记录中，有哪些具体的客户提到竞争对手的品比我们的价格低的？
                    请你列举出具体的客户名和商品名，以及竞争对手的名称和价格（如果有的话）。
                    **请你完全基于CSV的数据做分析，如果用户没有具体的反馈内容，请不要推测。这对公司来说非常重要，我们需要使用真实的客户反馈去调整经营策略**
                    以下是拜访记录CSV内容：\n\n{csv_string}""",
                    }
                ],
            },
        ],
    )

    merged_markdown_result = (
        f"{merged_markdown_result}## 竞争对手情况分析\n\n{text}\n\n"
    )

    text, is_ok = call_azure_openai(
        is_gpt4o=False,
        json=False,
        messages=[
            system_prompt,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""作为销售主管，请你分析客户长时间未下单的原因中，有哪些值得注意的现象？请你列举每种原因的占比，凸显出每个原因的重要程度。
                    **请你完全基于CSV的数据做分析，如果用户没有反馈具体的原因，请不要推测。这对公司来说非常重要，我们需要使用真实的客户反馈去调整经营策略**
                    以下是拜访记录CSV内容：\n\n{csv_string}""",
                    }
                ],
            },
        ],
    )

    merged_markdown_result = (
        f"{merged_markdown_result}## 长时间不下单原因分析\n\n{text}\n\n"
    )

    filename = f"{city}_销售团队拜访记录分析结果_{ds}.md"

    # Full path for the output file
    output_path = os.path.join(output_dir, filename)

    # Write the merged_markdown_result to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged_markdown_result)

    print(f"Markdown file saved: {output_path}")


# In[ ]:


# Display all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

bd_follow_up_record_df=merged_df

# bd_follow_up_record_df['communication_time_in_seconds']=bd_follow_up_record_df['communication_time_in_seconds'].astype(int)
# bd_follow_up_record_df.drop_duplicates(subset=["sessionid"], inplace=True)


# In[ ]:


import json

bd_follow_up_record_df = bd_follow_up_record_df.dropna(subset=["cust_id"])
bd_follow_up_record_df.to_csv(f"{output_dir}/bd_follow_up_record_df.csv", index=False)

bd_follow_up_record_df.rename(
    columns={
        "conversation_list": "拜访内容_segments",
        "bd_name": "拜访人",
    },
    inplace=True,
    errors="ignore",
)


whisper_segments_csv = bd_follow_up_record_df[
    ["商户名", "拜访人", "是否下过单", "距离上次下单天数", "拜访内容_segments"]
]
whisper_segments_csv["拜访内容_segments"] = whisper_segments_csv[
    "拜访内容_segments"
].apply(lambda segments: ", ".join(segments) if isinstance(segments, list) else "")
whisper_segments_csv.to_csv(
    f"{output_dir}/拜访内容_企业微信.csv", index=False
)


# In[ ]:


# Assign default value for filtered rows
filter_condition = bd_follow_up_record_df["拜访内容_segments"].apply(
    lambda segments: all(
        "非文本" in segment or "您好，您的服务已升级" for segment in segments
    )
)

default_json_value = json.dumps(
    {
        "客户是否有跟销售员互动": "无,默认值",
        "销售向客户推荐了哪些具体的活动": "无",
        "销售向客户推荐了哪些具体的商品": "无",
        "客户的主要采买渠道": "无",
        "客户对公司的产品有什么看法": "无",
        "客户对公司的配送服务有什么看法": "无",
        "客户对公司的评价是怎样的": "无",
        "客户不愿意下单的原因": "无",
        "销售员本次拜访的主要目的": "无",
        "销售员解决了客户哪些问题": "无",
        "拜访记录完整性打分": 0,
    },
    ensure_ascii=False,
)


def create_ai_analytics_for_row(row):
    visit_text = f"对话条数{len(row['拜访内容_segments'])}条, 通话记录:{row['拜访内容_segments']}"
    print(f"{row['拜访人']}, visit_text:{visit_text}")
    segments = row["拜访内容_segments"]
    segments_with_text = [
        segment
        for segment in segments
        if "非文本" not in segment and "您好，您的服务已升级" not in segment
    ]
    if len(segments_with_text) == 0:
        print(f"{row['拜访人']}, 没有文本内容, 使用默认值")
        return default_json_value
    return call_ai_api_to_get_extract_visit_info(
        f"对话条数{len(segments)}条, 通话记录:{segments_with_text}"
    )


bd_follow_up_record_df["AI分析"] = bd_follow_up_record_df.apply(
    create_ai_analytics_for_row, axis=1
)


# In[ ]:


bd_follow_up_record_df['拜访记录完整性打分'] = bd_follow_up_record_df['AI分析'].apply(
    lambda x: json.loads(x).get('拜访记录完整性打分', '0')
)
bd_follow_up_record_df['拜访记录完整性打分'] = pd.to_numeric(bd_follow_up_record_df['拜访记录完整性打分'], errors='coerce')


# In[ ]:


# 为哪些打分大于等于60分的聊天记录单独进行'AI总结'


def create_ai_summary_for_row(row):
    if row["拜访记录完整性打分"] >= 60:
        return call_azure_openai(
            is_gpt4o=False,
            json=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""从以下销售员和客户之间的企业微信对话中，你发现了哪几个最值得公司管理者关注的问题？
请你结合客户的基本情况，以及销售员客户的聊天记录，列举出明确的值得管理层引起注意的内容，并且标注出处(聊天记录的原文)\n\n
**请记住：你不需要给出建议，只需要列出值得管理层引起注意的内容，并且标注出处**\n\n
客户距离上次下单天数:{row['距离上次下单天数']}, 客户历史总下单金额:¥{row['历史总下单金额']}, 对话条数:{len(row['拜访内容_segments'])}条\n\n\n拜访内容:\n{row['拜访内容_segments']}""",
                        }
                    ],
                },
            ],
        )[0]

    else:
        return f"拜访记录不够完整，无需AI总结, 打分:{row['拜访记录完整性打分']}, 内容:{row['拜访内容_segments']}"


bd_follow_up_record_df["AI总结"] = bd_follow_up_record_df.apply(
    create_ai_summary_for_row, axis=1
)

bd_follow_up_record_df.sort_values(
    by="拜访记录完整性打分", ascending=False, inplace=True
)


# In[ ]:


import json

keys = []


def extract_ai_result(ai_result, key):
    return json.loads(ai_result).get(key, "未知")


for sale_man_name in bd_follow_up_record_df["BD企业微信ID"].unique():
    logging.info(f"开始处理:{sale_man_name}的拜访记录")
    sale_man_df = bd_follow_up_record_df[
        bd_follow_up_record_df["BD企业微信ID"] == sale_man_name
    ].copy()

    # Create a valid filename by replacing any characters that might be problematic in filenames
    safe_city_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in sale_man_name
    )

    # Save the city's records to a CSV file
    filename = f"./{output_dir}/{safe_city_name}_{ds}_拜访记录.csv"
    sale_man_df[
        [
            "BD企业微信ID",
            "m1负责人",
            "商户名",
            "距离上次下单天数",
            "拜访记录完整性打分",
            "历史下单数",
            "历史总下单金额",
            "拜访内容_segments",
            "AI分析",
        ]
    ].to_csv(filename, index=False)

    print(f"Saved {len(sale_man_df)} records for {sale_man_name} to {filename}")

    for index, row in sale_man_df.iterrows():
        ai_result = json.loads(row["AI分析"])
        if not keys:
            keys = list(ai_result.keys())
            logging.info(f"keys: {keys}")
            break

    for key in keys:
        sale_man_df[key] = sale_man_df["AI分析"].apply(
            lambda x: extract_ai_result(x, key)
        )

    display_keys = [
        "销售区域",
        "BD企业微信ID",
        "m1负责人",
        "商户名",
        "距离上次下单天数",
        "拜访记录完整性打分",
        "历史下单数",
        "历史总下单金额",
        "AI分析",
        "AI总结",
    ]
    display_keys.extend(keys)
    sale_man_df[display_keys].to_csv(
        f"./{output_dir}/{safe_city_name}_{ds}_拜访记录_AI分析_展开.csv", index=False
    )

    ai_csv_analytics_keys = [
        "销售区域",
        "BD企业微信ID",
        "m1负责人",
        "商户名",
        "距离上次下单天数",
        "拜访记录完整性打分",
        "历史下单数",
        "历史总下单金额",
    ]
    ai_csv_analytics_keys.extend(keys)
    csv_string = sale_man_df[ai_csv_analytics_keys].to_csv(index=False)

    print(f"{sale_man_name}, \ncsv_string:{csv_string}")

    call_ai_api_to_get_insigns(csv_string=csv_string, city=safe_city_name)


# In[ ]:




