#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# Save merged_markdown_result to a markdown file
import os
from datetime import datetime,timedelta

ds = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

# Create a directory for the markdown files if it doesn't exist
output_dir = f"output_qiyu_voice{ds}"
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
**请你用问题的标题做JSON的key，答案做value，比如：{"客户是否有跟销售员互动": "是,交流了10句话"}**
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


# In[ ]:


import sys, os

# Expand the `~` to the full path and append it to `sys.path`
full_path = os.path.expanduser("~/Documents/github/aliyun-devops")
sys.path.append(full_path)

from odps_client import get_odps_sql_result_as_df
from datetime import datetime, timedelta

staffname_list = ["白津源", "宋懿航", "李梦婷", "陈汉文"]
staffname_list = [f"'{staffname}'" for staffname in staffname_list]
staffname_list = ",".join(staffname_list)

sql = f"""
SELECT  a.*
        ,CAST(b.cust_id AS BIGINT) cust_id
        ,b.cust_name 商户名
        ,"电话拜访" AS 拜访目的
        ,"电话拜访" AS 拜访类型
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
FROM    summerfarm_ds.crm_qiyu_call_feishu_result_raw_di a
LEFT JOIN summerfarm_tech.dim_cust_df b
ON      a.`to` = b.cust_phone
AND     b.ds = MAX_PT('summerfarm_tech.dim_cust_df')
LEFT JOIN summerfarm_tech.dim_bd_df c
ON      c.bd_name = a.staffname
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
WHERE   a.ds = '{ds}'
and a.staffname in ({staffname_list})
;
"""
print(f"sql:{sql}")
bd_follow_up_record_df = get_odps_sql_result_as_df(sql=sql)
bd_follow_up_record_df.drop_duplicates(
    subset=["sessionid"], inplace=True
)


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

bd_follow_up_record_df['communication_time_in_seconds']=bd_follow_up_record_df['communication_time_in_seconds'].astype(int)
bd_follow_up_record_df.drop_duplicates(subset=["sessionid"], inplace=True)


# In[ ]:


# 下载wav文件到本地，用于whisper模型识别

from concurrent.futures import ThreadPoolExecutor
import concurrent
import requests
import os


def download_wav_file(url, sessionid, communication_time_in_seconds=0):
    logging.info(f"下载语音文件:{url}, session:{sessionid}, 通话时长:{communication_time_in_seconds}s")
    file_name = os.path.basename(url)
    response = requests.get(url)
    local_file = f"{output_dir}/{sessionid}_{file_name}"
    if os.path.exists(local_file):
        logging.info(f"The file {local_file} already exists.")
        return

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary mode to write the content
        with open(local_file, "wb") as f:
            f.write(response.content)
        logging.info(f"File {file_name} downloaded successfully.")
    else:
        logging.info("Failed to download the file.")


with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(download_wav_file, row["recordurl"], row["sessionid"], row['communication_time_in_seconds'])
        for index, row in bd_follow_up_record_df.iterrows()
    ]
    concurrent.futures.wait(futures)


# In[ ]:


def call_whisper_with_wav_file(row):
    logging.info(
        f"请求Groq进行语音识别, 文件:{row['recordurl']}, session:{row['sessionid']}, 通话时长:{row['communication_time_in_seconds']}s"
    )
    if int(row["communication_time_in_seconds"]) <= 10:
        error_text = (
            f"通话时长只有{row['communication_time_in_seconds']}s, 无需进行语音识别"
        )
        return {
            "text": error_text,
            "segments": error_text,
        }
    file_name = os.path.basename(row["recordurl"])
    local_file = f"{output_dir}/{row['sessionid']}_{file_name}"
    if not os.path.exists(local_file):
        logging.info(f"The file {local_file} does not exists.")
        return f"File not exists:{local_file}"

    return call_whisper_with_local_file(local_file)


import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

data = {
    "model": "whisper-large-v3",
    "language": "zh",
    "response_format": "verbose_json",
}


def call_whisper_with_local_file(
    filename="/Users/tangpeng/Downloads/白津源_3d341fed6a4637ad614830dc9d1a6b97.wav",
):
    with open(filename, "rb") as audio_file:
        files = {"file": (audio_file.name, audio_file)}

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
        }

        response = requests.post(
            groq_api_url,
            headers=headers,
            data=data,
            files=files,
            proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},
        )

        if response.status_code == 200:
            transcription = response.json()
            print(f"Transcription:{transcription}")  # Example access
            return {
                "text": "\n".join(
                    [segment["text"] for segment in transcription["segments"]]
                ),
                "segments": [
                    f'{segment["start"]}s-{segment["end"]}s: {segment["text"]}'
                    for segment in transcription["segments"]
                ],
            }
        else:
            print("Error:", response.status_code, response.text)
            return None


bd_follow_up_record_df[["whisper_recongnize_text", "whisper_recongnize_segments"]] = (
    bd_follow_up_record_df.apply(call_whisper_with_wav_file, axis=1).apply(pd.Series)
)


# In[ ]:


import json

bd_follow_up_record_df = bd_follow_up_record_df.dropna(subset=["商户名"])
bd_follow_up_record_df.to_csv(f"{output_dir}/bd_follow_up_record_df.csv", index=False)

bd_follow_up_record_df.rename(
    columns={
        "feishu_file_recognize_result": "拜访内容_飞书",
        "whisper_recongnize_text": "拜访内容_whisper",
        "whisper_recongnize_segments": "拜访内容_segments",
        "communication_time_in_seconds": "通话时长s",
        "staffname": "拜访人",
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
    f"{output_dir}/拜访内容_飞书_whisper_segments.csv", index=False
)


# In[ ]:


bd_follow_up_record_df["AI分析"] = bd_follow_up_record_df.apply(
    lambda row: call_ai_api_to_get_extract_visit_info(
        f"通话时长{row['通话时长s']}s, 通话记录:{row['拜访内容_segments']}"
    ),
    axis=1,
)


# In[ ]:


# Extract '拜访记录完整性打分' from 'AI分析' column
bd_follow_up_record_df['拜访记录完整性打分'] = bd_follow_up_record_df['AI分析'].apply(
    lambda x: json.loads(x).get('拜访记录完整性打分', '未知')
)

# Convert to numeric, replacing '未知' with NaN
bd_follow_up_record_df['拜访记录完整性打分'] = pd.to_numeric(bd_follow_up_record_df['拜访记录完整性打分'], errors='coerce')


# In[ ]:


# Create a new column 'AI总结' by calling the Azure OpenAI API for each row
bd_follow_up_record_df["AI总结"] = bd_follow_up_record_df.apply(
    lambda row: (
        "拜访记录不够完整，无需AI总结"
        if row["拜访记录完整性打分"] < 60
        else call_azure_openai(
            is_gpt4o=False,
            json=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""从以下销售员和客户之间的通话录音中，你发现了哪几个最值得公司管理者关注的问题？
请你结合客户的基本情况，以及销售员拜访的记录，列举出明确的值得管理层引起注意的内容，并且标注出处(发生在录音内容的哪一个时段)\n\n
你不需要给出建议，只需要列出值得管理层引起注意的内容，并且标注出处\n\n
客户距离上次下单天数:{row['距离上次下单天数']}, 客户历史总下单金额:¥{row['历史总下单金额']}, 通话时长:{row['通话时长s']}s\n拜访内容:\n{row['拜访内容_segments']}""",
                        }
                    ],
                },
            ],
        )[0]
    ),
    axis=1,
)


# In[ ]:


bd_follow_up_record_df.sort_values(by="拜访记录完整性打分", ascending=False, inplace=True)
# bd_follow_up_record_df[["AI总结","AI分析","拜访记录完整性打分","拜访内容_whisper"]].head(20)


# In[ ]:


import json

keys = []


def extract_ai_result(ai_result, key):
    return json.loads(ai_result).get(key, "未知")


for city in bd_follow_up_record_df["拜访人"].unique():
    logging.info(f"开始处理:{city}的拜访记录")
    sale_man_df = bd_follow_up_record_df[
        bd_follow_up_record_df["拜访人"] == city
    ].copy()

    # Create a valid filename by replacing any characters that might be problematic in filenames
    safe_city_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in city)

    # Save the city's records to a CSV file
    filename = f"./{output_dir}/{safe_city_name}_{ds}_拜访记录.csv"
    sale_man_df[
        [
            "拜访人",
            "m1负责人",
            "商户名",
            "距离上次下单天数",
            "拜访记录完整性打分",
            "历史下单数",
            "历史总下单金额",
            "拜访内容_whisper",
            "AI分析",
            "recordurl",
        ]
    ].to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"Saved {len(sale_man_df)} records for {city} to {filename}")

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
        "拜访人",
        "m1负责人",
        "商户名",
        "距离上次下单天数",
        "拜访记录完整性打分",
        "历史下单数",
        "历史总下单金额",
        "拜访内容_whisper",
        "AI分析",
        "AI总结",
    ]
    display_keys.extend(keys)
    sale_man_df[display_keys].to_csv(
        f"./{output_dir}/{safe_city_name}_{ds}_拜访记录_AI分析_展开.csv", index=False
    )

    ai_csv_analytics_keys = [
        "销售区域",
        "拜访人",
        "m1负责人",
        "商户名",
        "距离上次下单天数",
        "拜访记录完整性打分",
        "历史下单数",
        "历史总下单金额",
    ]
    ai_csv_analytics_keys.extend(keys)
    csv_string = sale_man_df[ai_csv_analytics_keys].to_csv(index=False)

    print(f"{city}, \ncsv_string:{csv_string}")
    call_ai_api_to_get_insigns(csv_string=csv_string, city=safe_city_name)


# In[ ]:





# In[ ]:




