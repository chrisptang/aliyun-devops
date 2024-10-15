#!/usr/bin/env python
# coding: utf-8

# ## 文件头，准备各类API

# In[ ]:


import os
import json
from typing import List
import shutil

from alibabacloud_sls20201230.client import Client as Sls20201230Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_sls20201230 import models as sls_20201230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
import os
import pandas as pd
from odps import ODPS, DataFrame
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import argparse
import logging

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ALIBABA_CLOUD_ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
FEISHU_XGPT_APP_SECRET = os.getenv("FEISHU_XGPT_APP_SECRET")
AZURE_GPT4O_API_KEY = os.getenv("AZURE_GPT4O_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
CALL_AI_SERVICE = os.getenv("CALL_AI_SERVICE", "true")

ds_to_run = datetime.now().strftime("%Y-%m-%d 00:00:00")
ds_to_run = datetime.strptime(ds_to_run, "%Y-%m-%d 00:00:00") - timedelta(days=1)
ds_to_run = ds_to_run.strftime("%Y%m%d")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_to_run",
    default=ds_to_run,
    help="指定跑哪一天的数据，格式: 20250520",
)
parser.add_argument(
    "--ACCESS_KEY_ID",
    default=ALIBABA_CLOUD_ACCESS_KEY_ID,
    help="ALIBABA_CLOUD_ACCESS_KEY_ID",
)
parser.add_argument(
    "--ACCESS_KEY_SECRET",
    default=ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    help="ALIBABA_CLOUD_ACCESS_KEY_SECRET",
)
parser.add_argument(
    "--FEISHU_XGPT_APP_SECRET",
    default=FEISHU_XGPT_APP_SECRET,
    help="飞书的FEISHU_XGPT_APP_SECRET",
)
parser.add_argument(
    "--AZURE_API_KEY",
    default=AZURE_API_KEY,
    help="AZURE_API_KEY",
)
parser.add_argument(
    "--AZURE_GPT4O_API_KEY", default=AZURE_GPT4O_API_KEY, help="AZURE_GPT4O_API_KEY"
)

args, unknown = parser.parse_known_args()

logging.info(f"Parsed args: {args}")
logging.info(f"Unknown args: {unknown}")
logging.info(args)
ds_to_run = args.ds_to_run

DATA_PATH = f"./data/{ds_to_run}"

logging.info(f"ds_to_run:{ds_to_run}")

default_segment_duration = int(os.getenv("SEGMENT_DURATION", "45"))

odps = ODPS(
    args.ACCESS_KEY_ID,
    args.ACCESS_KEY_SECRET,
    project="summerfarm_ds_dev",
    endpoint="http://service.cn-hangzhou.maxcompute.aliyun.com/api",
)

config = open_api_models.Config(
    access_key_id=args.ACCESS_KEY_ID,
    access_key_secret=args.ACCESS_KEY_SECRET,
)
# Endpoint 请参考 https://api.aliyun.com/product/Sls
config.endpoint = f"cn-hangzhou.log.aliyuncs.com"
sls_client = Sls20201230Client(config)


def create_dir_if_not_exist(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


create_dir_if_not_exist(DATA_PATH)


def get_odps_sql_result_as_df(sql):
    logging.info(f"ODPS SQL:\n{sql}")
    instance = odps.execute_sql(
        sql,
        hints={"odps.sql.hive.compatible": True, "odps.sql.type.system.odps2": True},
    )
    instance.wait_for_success()
    pd_df = None
    with instance.open_reader(tunnel=True) as reader:
        # type of pd_df is pandas DataFrame
        pd_df = reader.to_pandas()

    if pd_df is not None:
        logging.info(f"columns:{pd_df.columns}")
        return pd_df
    return None


def add_new_column_to_table(table_name, column_name):
    if "summerfarm_ds." not in table_name:
        table_name = f"summerfarm_ds.{table_name}"
    sql = f"ALTER TABLE {table_name} ADD COLUMNS ({column_name} STRING);"
    instance = odps.execute_sql(sql)
    instance.wait_for_success()
    logging.info(f"添加新字段成功:{table_name}, {column_name}")


def ensure_all_df_columns_in_odps_table(df, table_name):
    if "summerfarm_ds." not in table_name:
        table_name = f"summerfarm_ds.{table_name}"
    if not odps.exist_table(table_name):
        logging.info(f"表不存在:{table_name}")
        return True
    table = odps.get_table(table_name)
    column_names = set([column.name for column in table.table_schema])
    column_names_out = ",\n".join(column_names)
    logging.info(f"DaraFrame字段合集:\n\n{column_names_out}")
    df_columns = df.columns.tolist()
    for df_col in df_columns:
        df_col = df_col.lower()
        if df_col not in column_names:
            logging.info(f"新字段:{df_col}, ODPS全部的字段:{column_names}")
            add_new_column_to_table(table_name, df_col)
    return True


def write_pandas_df_into_odps_overwrite(df, table_name, partition_spec):
    if df is None or len(df) <= 0:
        logging.info(f"数据DF为空, table:{table_name}")
        return False
    ensure_all_df_columns_in_odps_table(df, table_name)
    exception = None
    try:
        odps_df = DataFrame(df)
        odps_df.persist(
            table_name,
            partition=partition_spec,
            drop_partition=False,
            create_partition=True,
            overwrite=True,
            lifecycle=365,
        )
        logging.info(f"成功写入odps:{table_name}, partition_spec:{partition_spec}")
        return True
    except Exception as e:
        exception = e
        logging.info(f"写入ODPS不成功:{table_name}", e)
        raise exception


# ## 飞书接口认证

# In[ ]:


import base64
import wave
import requests
import ffmpeg
import re

def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
    token = requests.post(
        url=url,
        json={"app_id": "cli_a450bff26fbbd00d", "app_secret": args.FEISHU_XGPT_APP_SECRET},
    ).json()
    logging.info(token)
    return token


feishu_token = get_feishu_token()
feishu_headers_with_token = {
    "Authorization": f'Bearer {feishu_token["tenant_access_token"]}',
    "Content-Type": "application/json",
}

logging.info(feishu_headers_with_token)


# ## wav文件下载和识别接口

# In[ ]:


import ffmpeg
import os
import base64
import requests
import logging
import time
import string
import random

text_map = {}
failed_wav_files={}

def generate_random_string(length=16):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))

random_string = generate_random_string()
print(random_string)

def dump_failed_json_to_file(json_object, file_id):
    with open(f"{DATA_PATH}/failed_{file_id}.json", "w") as f:
        json.dump(json_object, f, indent=4)


def get_file_id_from_base64_content(s, num=16):
    # return "".join(re.findall(r"[a-zA-Z0-9]", s))[:num]
    return generate_random_string(num)


def get_dir_from_path(file_path):
    return os.path.dirname(file_path)


def split_wav_file(input_file, segment_duration=default_segment_duration):
    output_files = []
    basename, _ = os.path.splitext(os.path.basename(input_file))

    try:
        output_dir = get_dir_from_path(input_file)
        # Open the input file
        stream = ffmpeg.input(input_file)

        # Set the output format and codec
        stream = ffmpeg.output(
            stream,
            os.path.join(output_dir, f"{basename}_segment_%03d.wav"),
            codec="copy",
            f="segment",
            segment_time=segment_duration,
        )

        # Run the FFmpeg command
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        # Get the list of output files
        output_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith(f"{basename}_segment_") and f.endswith(".wav")
        ]
    except Exception as e:
        logging.error(f"切分wav文件:{input_file}出错了:{e}", e)

    output_files.sort()
    return output_files


def get_base64encoded_content_of_wav(wav_file):
    logging.info(f"wav_file:{wav_file}")

    output = ""

    # Create the input stream
    input_stream = ffmpeg.input(wav_file)

    # Configure the output stream
    output_stream = input_stream.output(
        "pipe:", acodec="pcm_s16le", format="s16le", ac=1, ar=16000
    )

    # Run the FFmpeg command and capture the output
    output, _ = ffmpeg.run(output_stream, capture_stdout=True, quiet=True)

    # The output is a bytes object containing the raw PCM data
    pcm_data = output
    return base64.b64encode(pcm_data).decode("utf-8")


def recognize_feishu_api_base64_content(base64_pcm_data, wav_file, retry=True):
    file_id = get_file_id_from_base64_content(base64_pcm_data)
    feishu_json = {
        "config": {
            "engine_type": "16k_auto",
            "file_id": file_id,
            "format": "pcm",
        },
        "speech": {"speech": f"{base64_pcm_data}"},
    }

    text = requests.post(
        "https://open.feishu.cn/open-apis/speech_to_text/v1/speech/file_recognize",
        json=feishu_json,
        headers=feishu_headers_with_token,
    ).text
    logging.info(f"飞书text:{text}")
    try:
        text = json.loads(text)
        return text["data"]["recognition_text"]
    except Exception as e:
        logging.info(f"调用飞书接口错误wav_file:{wav_file}, 接口返回:{text}, 异常信息:{e}")
        if retry:
            logging.warning("20s后重试1次:")
            time.sleep(20)
            return recognize_feishu_api_base64_content(
                base64_pcm_data=base64_pcm_data, wav_file=wav_file, retry=False
            )
        else:
            global failed_wav_files
            dump_failed_json_to_file(feishu_json, file_id)
            failed_wav_files['wav_file']=file_id
            logging.error(f"不再重试:{retry}")
        return text


def is_feishu504_error(api_text):
    return "504 Gateway Time-out" in api_text


def file_recognize_feishu_api(wav_file):
    # 先切分成2分钟一段：
    sub_files = split_wav_file(wav_file)

    if sub_files is None or len(sub_files) < 0:
        logging.info(f"切分失败：{wav_file}")
        return
    if len(sub_files) > 1:
        logging.info(f"切分成了多个小文件：{','.join(sub_files)}")
    all_text = []
    for file in sub_files:
        base64_pcm_data = get_base64encoded_content_of_wav(wav_file)
        logging.info(f"wav_file:{file}, base64_pcm_data: {base64_pcm_data[-50:]}")
        all_text.append(
            recognize_feishu_api_base64_content(
                base64_pcm_data=base64_pcm_data, wav_file=file
            )
        )
    return "".join(all_text)


# ## 获取ODPS数据

# In[ ]:


recordurl_df=get_odps_sql_result_as_df(f"""
SELECT  *,DATEDIFF(CAST(endtime AS TIMESTAMP),CAST(createtime AS TIMESTAMP),'ss') communication_time_in_seconds
FROM    (
            SELECT  JSON_TUPLE(body,"eventtype","sessionid","direction","createtime","endtime","connectionbeginetime","connectionendtime","from","to","user","category","staffid","staffname","status","visittimes","duration","evaluation","recordurl","overflowFrom","shuntGroupName","ivrPath","mobileArea","waitDuration","ringDuration","sessionIdFrom","firstEndDirection") AS ("eventtype","sessionid","direction","createtime","endtime","connectionbeginetime","connectionendtime","from","to","user","category","staffid","staffname","status","visittimes","duration","evaluation","recordurl","overflowFrom","shuntGroupName","ivrPath","mobileArea","waitDuration","ringDuration","sessionIdFrom","firstEndDirection")
            FROM    summerfarm_tech.ods_qiyu_call_log_di
            WHERE   ds = '{ds_to_run}'
            AND     GET_JSON_OBJECT(body,'$.eventtype') = '5'
        ) 
WHERE   recordurl LIKE 'https://hzxmkjyxgs7.%';""")

logging.info(f"数据量大小:{len(recordurl_df)}")


# In[ ]:


import requests
import os


def download_wav_file(url, sessionid, communication_time_in_seconds=0):
    # Send a request to the URL
    if communication_time_in_seconds<=30:
        logging.info(f"通话时长过短，不需下载语音文件:{url}, session:{sessionid}, 通话时长:{communication_time_in_seconds}s")
        return
    logging.info(f"下载语音文件:{url}, session:{sessionid}")
    file_name = os.path.basename(url)
    response = requests.get(url)
    local_file = f"{DATA_PATH}/{sessionid}_{file_name}"
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
        for index, row in recordurl_df.iterrows()
    ]
    concurrent.futures.wait(futures)


# ## 请求azure进行语音分析，还原对话

# In[ ]:


# Import the base64 encoding library.
import base64

proxy_object = {"http": "http://127.0.0.1:8001", "https": "http://127.0.0.1:8001"}


from openai import AzureOpenAI

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://xm-ai.openai.azure.com",
    api_key=args.AZURE_API_KEY,
)

client_gpt4o = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint="https://xm-ai-us2.openai.azure.com",
    api_key=args.AZURE_GPT4O_API_KEY,
)


def call_azure_openai(content="", command="", retrying=1, is_gpt4o=False) -> str:
    if retrying < 0:
        return "超过了最大重试次数", False
    completion = None
    ## gpt3.5:  gpt-35-turbo-16k,
    ## got4o:   gpt-4o
    model = "gpt-35-turbo-16k"
    client_to_use = client
    if is_gpt4o:
        logging.info(f"using GPT-4o...:{command}")
        model = "gpt-4o"
        client_to_use = client_gpt4o
    try:
        completion = client_to_use.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=4095,
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个资深的销售主管。\n**请你根据用户给的内容分析销售员和客户之间的对话。**\n通常来说对话都是销售员发起的。\n{command}",
                },
                {
                    "role": "user",
                    "content": f"```{content}```",
                },
            ],
        )
        response = completion.choices[0].message.content
        if (
            len(completion.choices) <= 0
            or f"{completion.choices[0].finish_reason}" == "content_filter"
        ):
            return f"azure过滤了本次请求:{completion.choices[0].to_dict()}", False
        if response is None:
            logging.info(f"azure API返回了异常:{completion.to_dict()}")
            return call_azure_openai(
                content=content,
                command=command,
                retrying=retrying - 1,
                is_gpt4o=is_gpt4o,
            )
        return response, True
    except Exception as e:
        logging.info(f"请求azure接口报错了:{e}\n content:{content}, completion:{completion}")
        if retrying <= 0 or "Error code: 400" in f"{e}":
            return f"{e}", False
    logging.info(f"重试中...{retrying}, content:{content}")
    return call_azure_openai(
        content=content, command=command, retrying=retrying - 1, is_gpt4o=is_gpt4o
    )


commands = [
    {
        "name": "对话总结",
        "text": "以下文本是我公司销售员/客服和客户之间的通话录音。请你总结对话的内容，提炼出核心事件",
    },
    {
        "name": "客情判断",
        "text": "以下文本是我公司销售员/客服和客户之间的通话录音。请你分析客户对我司的评价是正面的，还是负面的，并给出分析依据",
    },
    {
        "name": "销售达成情况",
        "text": "请你分析销售员与客户之间的对话内容，并判断我们的销售员是否达成了销售目标。如果客户问到了具体价格且表示认可，也算是销售成功。\n如果销售达成，则列出销售成功的商品名字。否则列出销售失败的原因",
    },
]


def call_ai_api_to_get_insigns(feishu_text):
    result = {}
    dialog, is_ok = call_azure_openai(
        is_gpt4o=True,
        content=feishu_text,
        command="""以下文本是我司销售员和客户之间的对话。
请你对对话内容进行还原，区分哪些内容是销售员说的、哪些是客户说的。
将销售员说的内容用"销售员："表示。将客户说的内容用"客户："表示。

## 请注意：
- **请你完全基于我给出的内容进行还原，未出现的对话内容不要出现在你的结果中。**
- **如果对话内容过于简短，比如少于50字，请你直接回复“对话内容过于简短，无需还原”**""",
    )
    result["对话还原"] = dialog
    if not is_ok:
        logging.info("失败:", dialog)
        return result
    if dialog is None or len(dialog) <= 30 or "对话内容过于简短，无需还原" in dialog:
        result["error"] = (
            f"AI返回的对话内容太短了，疑似出错了:{dialog}, 飞书文本:{feishu_text}"
        )
        logging.error(f"{result}")
        return result
    for command in commands:
        result[command["name"]], is_ok = call_azure_openai(dialog, command["text"])
    return result


# In[ ]:


def get_feishu_file_recognize_result_for_row(row_dict):
    try:
        row = row_dict
        communication_time_in_seconds = row["communication_time_in_seconds"]
        if communication_time_in_seconds <= 30:
            ignored = f"通话时长不到30s:{communication_time_in_seconds}s,{row['user']}, {row['sessionid']}"
            row["feishu_file_recognize_result"] = ignored
            return ignored
        logging.info(f"sessionid:{row['sessionid']}, recordurl:{row['recordurl']}")
        text = ""
        text = file_recognize_feishu_api(
            f"{DATA_PATH}/{row['sessionid']}_{os.path.basename(row['recordurl'])}"
        )
        logging.info(f"{row['sessionid']} 的文本:{text}")
        row["feishu_file_recognize_result"] = text
        return text
    except Exception as e:
        row["feishu_file_recognize_result"] = f"错误:{e}"
        return "ERROR"


row_dict_list = [row.to_dict() for _, row in recordurl_df.iterrows()]

with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit tasks to the executor
    futures = [
        executor.submit(get_feishu_file_recognize_result_for_row, row_dict)
        for row_dict in row_dict_list
    ]
    concurrent.futures.wait(futures)

recordurl_with_text_df = pd.DataFrame(row_dict_list)

recordurl_with_text_df[["sessionid", "user", "staffname", "feishu_file_recognize_result"]]


# ## 写入ODPS

# In[ ]:


def get_gemini_result(row):
    feishu_file_recognize_result = row["feishu_file_recognize_result"]
    logging.info(f"feishu_file_recognize_result: {feishu_file_recognize_result}")
    if (
        "504 Gateway Time-out" in f"{feishu_file_recognize_result}"
        or "通话时长不到30s" in f"{feishu_file_recognize_result}"
    ):
        logging.info(f"飞书文本异常：{feishu_file_recognize_result}")
        return feishu_file_recognize_result

    logging.info(
        f"sessionid:{row['sessionid']}, feishu_file_recognize_result:{feishu_file_recognize_result}"
    )
    ai_response = call_ai_api_to_get_insigns(feishu_file_recognize_result)
    logging.info(f"{row['sessionid']} API分析:\n{ai_response}")
    return f"{ai_response}"


recordurl_with_text_df.to_csv(
    f"{DATA_PATH}/qiyu_records_with_feishu_recognize_result.csv", index=False
)
# 先保存一份仅仅包含飞书语音识别的结果
write_pandas_df_into_odps_overwrite(
    recordurl_with_text_df.astype(str),
    "summerfarm_ds.crm_qiyu_call_feishu_result_raw_di",
    f"ds={ds_to_run}",
)

if "true" == CALL_AI_SERVICE:
    recordurl_with_text_df["gemini_result"] = recordurl_with_text_df.apply(
        get_gemini_result, axis=1
    )
    recordurl_with_text_df = recordurl_with_text_df.astype(str)
    recordurl_with_text_df.to_csv(
        f"{DATA_PATH}/qiyu_records_with_ai_result.csv", index=False
    )
    write_pandas_df_into_odps_overwrite(
        recordurl_with_text_df,
        "summerfarm_ds.crm_qiyu_call_analytics_raw_v2_di",
        f"ds={ds_to_run}",
    )
else:
    logging.warn("将不请求AI服务！！")

logging.info(f"成功了！\n>>>>>>>>>>>>\n失败的wav文件列表:{failed_wav_files}")


# In[ ]:





# In[ ]:




