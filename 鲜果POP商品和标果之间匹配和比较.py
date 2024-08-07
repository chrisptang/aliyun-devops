#!/usr/bin/env python
# coding: utf-8

# ## 登录admin

# In[ ]:


# 获取token
import requests
import os

login_url = "https://admin.summerfarm.net/authentication/auth/username/login"
login_data = {
    "username": "peng.tang@summerfarm.net",
    "password": os.getenv("XIANMU_ADMIN_PASSWORD"),
}

token = requests.post(login_url, data=login_data).json()

print(token)


headers = {
    "token": token.get("data").get("token"),
    "xm-rqid": "create_fake_merchant_tp",
    "xm-uid": "2047",
    "Content-Type": "application/json;charset=UTF-8",
}

print(headers)


# In[ ]:


def get_product_info(pd_id=11087):
    url = "https://admin.summerfarm.net/sf-mall-manage/product/query/info"
    data = {"pdId": pd_id}
    return requests.post(url=url, headers=headers, json=data).json().get("data")


# Define the 'CASE WHEN' logic
def open_sale_case_when(open_sale=-1):
    if open_sale == 0:
        return "上架"
    elif open_sale == 1:
        return "有库存时上架"
    elif open_sale == 2:
        return "定时上架"
    elif open_sale == 3:
        return "有库存时上架(永久生效)"
    else:
        return "未定义"
    
pd_list = (
    requests.get(
        "https://admin.summerfarm.net/sf-mall-manage/product/query/selectPage?grandCategoryId=1132&pageIndex=1&pageSize=500",
        headers=headers,
    )
    .json()
    .get("data")
    .get("list")
)


# ## 构建鲜沐的类目树

# In[ ]:


url = "https://admin.summerfarm.net/category"
all_xianmu_category = requests.get(url, headers=headers).json().get("data")
all_xianmu_category_pop = []
for cate in all_xianmu_category:
    if "POP" in cate["category"]:
        all_xianmu_category_pop = cate.get("categoryList")
        break
# print(all_xianmu_category_pop)
all_xianmu_category_pop_map = {}
for second_level in all_xianmu_category_pop:
    second_category_name = second_level.get("category", "")
    for leaf_level in second_level.get("categoryList", []):
        all_xianmu_category_pop_map[f"{leaf_level.get('id','haha')}"] = (
            f"{second_category_name}/{leaf_level['category']}"
        )

all_xianmu_category_pop_map


# ## 从鲜沐admin后台获取所有的POP商品

# In[ ]:


import json
import pandas as pd


def extract_product_fields(json_data):
    # Initialize a list to hold the extracted data
    extracted_data = []
    keyValueList = ",".join(
        [
            f"{kv['name']}:{kv['productsPropertyValue']}"
            for kv in json_data.get("keyValueList", [])
        ]
    )

    # Loop through each item in the inventory detail list
    for item in json_data.get("inventoryDetailVOS", []):
        # Concatenate saleValueList
        sale_value_list = item.get("saleValueList", [])
        buyer_name = item.get("buyerName", "")
        sale_value_string = ",".join(
            f"{sv['name']}:{sv['productsPropertyValue']}" for sv in sale_value_list
        )
        # Loop through each area SKU in the item
        for area_sku in item.get("areaSkuVOS", []):
            # Extract the necessary fields and create a dictionary
            extracted_dict = {
                "sku": area_sku.get("sku", ""),
                "categoryName": all_xianmu_category_pop_map.get(
                    f'{json_data.get("categoryId")}', "哈哈哈哈"
                ),
                "openSale": open_sale_case_when(area_sku.get("openSale", -1)),
                "onSale": "上架中" if area_sku.get("onSale", False) else "下架(没库存)",
                "pdId": item.get("pdId", ""),
                "sku_name": json_data.get("productName", ""),
                "pd_name": json_data.get("pdName", ""),
                "realName": item.get("realName", ""),
                "picture": "https://azure.summerfarm.net/"
                + json_data.get("picturePath", "404.jpg"),
                "price": area_sku.get("price", 0),
                "supplyPrice": json.loads(item.get("createRemark", "{}")).get(
                    "supplyPrice", 0
                ),
                "areaNo": area_sku.get("areaNo", ""),
                "areaName": area_sku.get("areaName", ""),
                "netWeightNum": item.get("netWeightNum", 0),
                "weightNum": item.get("weightNum", 0),
                "weight": item.get("weight", ""),
                "sku_spec": sale_value_string,
                "spu_spec": keyValueList,
                "buyer_name": buyer_name,
            }
            # Add the dictionary to the list
            extracted_data.append(extracted_dict)

    return extracted_data


# In[ ]:


product_list = []

for product in pd_list:
    product = get_product_info(product["pdId"])
    product_list.append(product)


# In[ ]:


from datetime import datetime, timedelta

sku_list = []

for product in product_list:
    extracted_data = extract_product_fields(product)
    sku_list.extend(extracted_data)

sku_list_df = pd.DataFrame(sku_list)
sku_list_df.to_csv(
    f"./data/鲜果POP全部SKU_{datetime.now().strftime('%Y%m%d')}.csv", index=False
)
sku_list_df.sort_values(by=["categoryName", "pd_name"], inplace=True)
sku_list_df.head(2)


# ### 获取标果的爬虫数据（当天）

# In[ ]:


# from odps_client import get_odps_sql_result_as_df

# all_fruit_category=['千禧樱桃小番茄','红富士苹果','水蜜桃','25号小蜜','冬枣','更多桃','麒麟西瓜','更多梨','更多苹果','更多李']
# all_fruit_category.extend(['柠檬','果篮','国产油桃','山竹','巨峰葡萄','阳光玫瑰','进口橙','凯特芒','国产红心火龙果','更多凤梨','更多提子'])
# all_fruit_category.extend(['进口红心火龙果','更多蜜瓜','蜂糖李','青脆李','更多柚子','水仙芒','玉菇甜瓜','百香果','更多榴莲','木瓜/杨桃','椰青'])
# all_fruit_category.extend(['金枕榴莲','人参果/释迦果','更多龙眼','黄桃','红肉菠萝蜜','三红蜜柚','台芒','吊干杏','无籽红提','更多葡萄','网纹蜜瓜','蜜桔'])
# all_fruit_category.extend(['赣南脐橙','黑布林','夏黑葡萄','普通红提','更多柑','蓝莓','西州蜜瓜','更多芒果','水果黄瓜','火腿肠','牛油果','硒砂瓜'])
# all_fruit_category.extend(['莲雾/芭乐','其他桔','圆红/血橙','更多橙','白心火龙果','皇冠梨','红布林','进口车厘子','更多荔枝','更多莓','桂味','特小凤','蛇果'])
# all_fruit_category.extend(['贡梨','贵妃芒','更多西瓜','白心蜜柚','秋月梨','绿心猕猴桃','羊角蜜','雪莲果/马蹄果','黄元帅苹果','冷冻畜禽食品','更多枣','水产品'])
# all_fruit_category.extend(['火参果','红心蜜柚','红毛丹','脆柿','软籽石榴','青芒','鲜山楂','鹰嘴芒'])

# biaoguo_df=get_odps_sql_result_as_df(f"""
# SELECT  categoryname,backcategoryname,id,competitor,skucode,spider_fetch_time,goodspropdetaillist,createtime,goodssiphoncommissionrate
#         ,standardprice,finalstandardprice,lasttimestandardprice,finalunitpricecatty,monthsale,attachurlr AS url,sellersiphoncommissionrate
#         ,unitpricecatty,unit,sellername,grossweight,netweight,specification,babyname,goodsname,goodstype,sevendayaftersale
# FROM    (
#             SELECT  *
#                     ,RANK() OVER (PARTITION BY id,skucode ORDER BY spider_fetch_time DESC ) AS rnk
#             FROM    summerfarm_ds.spider_biaoguo_with_prop_product_result_df
#             WHERE   ds = MAX_PT('summerfarm_ds.spider_biaoguo_with_prop_product_result_df')
#             AND     competitor = '标果-杭州' 
#                                      AND categoryname like '%/%'
#             -- AND     split_part(categoryname,'/',2) in ('{"','".join(all_fruit_category)}')
#         ) 
# WHERE   rnk = 1
# LIMIT   100000;
# """)

# print(f"标果的商品数:{len(biaoguo_df)}")
# biaoguo_df.head(1)


# In[ ]:


# print(f"标果的所有类目: \n{biaoguo_df['categoryname'].unique()}")

# category_list_that_xianmu_wont_sale = set(
#     [
#         "即食食品/方便罐头",
#         "即食食品/海苔食品",
#         "饼干糕点/饼干",
#         "方便食品/方便速食" "糖果/巧克力/果冻布丁/果冻/布丁",
#         "刀具工具/刀具工具",
#         "包装配饰/包装配饰",
#         "水果盒/带盖盒",
#         "陈列道具/陈列道具" "水果盒/托盒",
#         "保鲜膜/保鲜膜",
#         "标签贴纸/标签贴纸",
#         "糖果/巧克力/果冻布丁/龟苓膏/果膏" "即食食品/肉干肉铺",
#         "坚果炒货/坚果",
#         "其他标品/营养保健",
#         "糖果/巧克力/果冻布丁/其他糖果",
#         "饼干糕点/散装食品" "酒水/饮料/冲调",
#         "即食食品/其他即食",
#         "酒水/饮料/酒类",
#         "饼干糕点/糕点",
#         "标果奶品/标果奶品" "即食食品/蔬菜干/豆干",
#         "糖果蜜饯/果干蜜饯",
#         "即食食品/火腿肠",
#         "水果袋/水果袋" "氛围牌/氛围牌",
#         "设计定制/设计定制",
#         "水果盒/瓶罐",
#         "粽子/米类/面类/粽子/米类/面类",
#         "洗护用品/洗护用品",
#         "鲜花/鲜花",
#         "水产/水产品",
#         "即食食品/卤味即食",
#     ]
# )

# print(f"这些类目被过滤掉了:{category_list_that_xianmu_wont_sale}")

# # Fix the filtering by using 'isin' method

# biaoguo_df = biaoguo_df[
#     ~biaoguo_df["categoryname"].isin(category_list_that_xianmu_wont_sale)
# ].copy()

# # Get the length of the filtered DataFrame
# print(f"过滤后的标果商品个数:{len(biaoguo_df)}")


# ## Azure客户端定义

# In[ ]:


import os
from openai import AzureOpenAI
import httpx

client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://xm-ai.openai.azure.com",
    api_key=os.getenv("AZURE_API_KEY_XM",'please set:AZURE_API_KEY_XM'),
    http_client=httpx.Client(proxies={"http://": None, "https://": None}),
)


# ## 使用embedding模型做标题的匹配

# In[ ]:


import sqlite3
import os

# Ensure the directory exists
os.makedirs(os.path.expanduser('~/sqlite'), exist_ok=True)
# Path to the database
db_path = os.path.expanduser('~/sqlite/embeddings.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS embeddings (
    input_text TEXT PRIMARY KEY,
    embedding TEXT
)
"""
)


def get_embedding_directly_from_azure(input: str):
    embbed = client.embeddings.create(model="text-embedding-ada-002", input=input)
    return embbed.to_dict().get("data", [{}])[0].get("embedding")


def get_embedding(input_text):

    # Check if input text already exists in the database
    cursor.execute(
        "SELECT embedding FROM embeddings WHERE input_text = ?", (input_text,)
    )
    result = cursor.fetchone()

    if result:
        embedding = result[0]
        print(f"Found, return the embedding of input_text:{input_text}, {embedding[:50]}")
    else:
        print(f"Not found, call the OpenAI API to get the embedding:{input_text}")
        embedding = str(get_embedding_directly_from_azure(input_text))

        # Insert the new input text and embedding into the database
        cursor.execute(
            "INSERT INTO embeddings (input_text, embedding) VALUES (?, ?)",
            (input_text, embedding),
        )
        conn.commit()
    return embedding


# Example usage
input_text = "你好"
embedding = get_embedding(input_text)
print(json.loads(embedding))


# In[ ]:


def get_xianmu_embedding(row: pd.Series):
    sku_name = row["sku_name"]
    if sku_name is None or len(f"{sku_name}") <= 2:
        sku_name = row["pd_name"]
    # weight = row["weight"] DO NOT USE
    input_text = f"{sku_name}"
    return get_embedding(input_text=input_text)


def get_competitor_like_biaoguo_embedding(row: pd.Series):
    goodsname = row["goods_name"]
    # specification = row["specification"] # specification is not used
    return get_embedding(input_text=f"{goodsname}")


def get_category_embedding(category: str):
    return get_embedding(input_text=category)


# biaoguo_df["sku_embeddings"] = biaoguo_df.apply(get_competitor_like_biaoguo_embedding, axis=1)
# sku_list_df["sku_embeddings"] = sku_list_df.apply(get_xianmu_embedding, axis=1)

# biaoguo_df["category_embeddings"] = biaoguo_df['categoryname'].apply(get_category_embedding)
# sku_list_df["category_embeddings"] = sku_list_df['categoryName'].apply(get_category_embedding)


# In[ ]:


import concurrent
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import traceback
from scipy.spatial.distance import cosine


top_n = 20


def calculate_cosine_similarity(embedding1, embedding2):
    if "class 'str'" in f"{type(embedding1)}":
        embedding1 = json.loads(embedding1)
    if "class 'str'" in f"{type(embedding2)}":
        embedding2 = json.loads(embedding2)
    # print(len(embedding1), len(embedding2), type(embedding2))
    return 1 - cosine(embedding1, embedding2)

sku_list_with_matched=[]

def find_top_matches_for_xianmu(sku_row: pd.Series):
    global sku_list_with_matched
    # print(f"finding top match rows for:{sku_row['sku_name']}")
    similarities = []
    for index, biaoguo_row in biaoguo_df.iterrows():
        # print(f"matching: {biaoguo_row['goodsname']}")
        similarity_score = calculate_cosine_similarity(
            sku_row["sku_embeddings"], biaoguo_row["sku_embeddings"]
        )
        category_similarity_score = calculate_cosine_similarity(
            sku_row["category_embeddings"], biaoguo_row["category_embeddings"]
        )
        # 类目embedding + sku embedding分数
        similarities.append(
            (biaoguo_row.to_dict(), similarity_score + category_similarity_score)
        )

    # Sort the results based on similarity score in descending order
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Get the top N matches
    top_n_matches = [item[0] for item in sorted_similarities[:top_n]]
    print(
        f"Matched rows for:{sku_row['sku_name']}, {[biaoguo.get('goodsname') for biaoguo in top_n_matches]}"
    )
    sku_row["top_matches"] = top_n_matches
    sku_list_with_matched.append(sku_row)
    return top_n_matches


# ### 写入到HTML文件

# In[ ]:


from IPython.core.display import HTML

css = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<style type=\"text/css\">
table {
    color: #333;
    font-family: unset;
    font-size: 12px;
    line-height: 1.5;
    width: 1024px;
    border-collapse:
    collapse; 
    border-spacing: 0;
    font-family: "SF Pro SC", "SF Pro Text", "SF Pro Icons", "PingFang SC", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
}

tr{
    border-bottom: 1px solid #C1C3D1;
}

tr:nth-child(even) {
    background-color: #F8F8F8;
}

td, th {
    /* border: 1px solid transparent; No more visible border */
    height: 30px;
}

th {
    background-color: #DFDFDF; /* Darken header a bit */
    font-weight: bolder;
    min-width: 100px;
    text-align: center;
}

td {
    /* background-color: #FAFAFA;
    text-align: center; */
}

ol li{
    text-align: left;
}
.biaoguo-container {
    display:flex;
}
.biaoguo-goods-item{
    padding-right: 0.3vw;
}
.no-list-style ul{
    padding-left: 0;
}
.no-list-style li{
    list-style: none;
    font-size: smaller;
}
.no-list-style li.sku-title{
    font-weight:bolder;
}
.xianmu-container{
    /* border-right: 1px solid lightblue; */
}
</style>
"""


def display_xianmu_html(row):
    img = row["picture"]
    unit_price = round(row["price"] / row.get("weightNum", 0) / 2, 2)
    content = f"""<div class="xianmu-container no-list-style">
    <img width="80" src="{img}">
    <ul>
    <li class="sku-title">{row['sku']}, {row["sku_name"]}</li>
    <li>鲜沐售价: ¥{row["price"]}</li>
    <li>单价(斤,毛重): ¥{unit_price}</li>
    <li>提报价: ¥{row['supplyPrice']}</li>
    <li>类目:{row.get('categoryName','')}</li>
    <li>规格:{row["weight"]}</li>
    <li>毛重:{row.get('weightNum', 0)*2}斤, 净重:{row.get('netWeightNum', 0)*2}斤</li>
    <li>买手:{row["buyer_name"]}</li>
    <li>是否上架:{row["onSale"]}</li>
    </ul></div>"""
    return content.replace("\n", "")


display_count = 10


def display_matched_biaoguo_html(row):
    top_matches = row["top_matches"]
    contents_of_single_item = []
    for index, item in enumerate(top_matches):
        if index >= display_count:
            # print(f"超过了配置数量:{display_count},跳过")
            break
        img = item["url"]
        finalunitpricecatty = round(
            float(item["finalstandardprice"]) / float(item["grossweight"]), 2
        )

        single_item = f"""<div class="biaoguo-goods-item no-list-style">
        <img width="80" src="{img}">
        <ul>
        <li class="sku-title">{item['skucode']}, {item['goodsname']}, {item['specification']}</li>
        <li>标果价格: ¥{item['finalstandardprice']}</li>
        <li>单价(斤,毛重): ¥{finalunitpricecatty}</li>
        <li>类目:{item['categoryname']}</li>
        <li>毛重:{item['grossweight']}斤, 净重:{item['netweight']}斤</li>
        <li>卖家:{item.get('sellername','--')}</li>
        <li>商品抽佣:{item.get('goodssiphoncommissionrate',0)}%</li>
        <li>向卖家抽佣:{item.get('sellersiphoncommissionrate',0)}%</li>
        <li>月销量:{item.get('monthsale',0)}, 7日售后:{item.get('sevendayaftersale',0)}</li>
        </ul></div>"""
        contents_of_single_item.append(single_item)

    return f"""<div class="biaoguo-container">{''.join(contents_of_single_item)}</div>""".replace(
        "\n", ""
    )

date_of_now = datetime.now().strftime("%Y-%m-%d")

# sku_list_df["鲜沐POP商品"] = sku_list_df.apply(display_xianmu_html, axis=1)
# sku_list_df["标果商品(Top10)"] = sku_list_df.apply(display_matched_biaoguo_html, axis=1)

# html_content = css + sku_list_df[["鲜沐POP商品", "标果商品(Top10)"]].to_html(
#     escape=False, index=False, classes="table dataframe"
# )
# html_content = f'<html><head><meta charset="UTF-8"><meta name="title" content="POP商品和标果商品的比较-{date_of_now}"></head><body>{html_content}</body></html>'

# file_path = f"./data/pop/鲜沐POP和标果-杭州的比较-{date_of_now}.html"

# # 保存HTML到本地文件：
# with open(file_path, "w", encoding="utf-8") as f:
#     f.write(html_content)

# print(f"写入HTML成功！{file_path}")


# ### 写入CSV，供大家参考

# In[ ]:


# sku_list_for_csv = []
# for _, row in sku_list_df.iterrows():
#     csv_object = {}
#     unit_price = round(row["price"] / row.get("weightNum", 0) / 2, 2)
#     xianmu_content = f"""{row['sku']}, {row["sku_name"]}
# 鲜沐售价: ¥{row["price"]}, 单价(斤,毛重): ¥{unit_price}
# 类目:{row.get('categoryName','')}
# 规格:{row["weight"]}
# 毛重:{row.get('weightNum', 0)*2}斤, 净重:{row.get('netWeightNum', 0)*2}斤"""
#     csv_object["顺鹿达SKU"] = xianmu_content

#     top_matches = row["top_matches"]
#     for i in range(0, 20):
#         if i >= len(top_matches):
#             break
#         item = top_matches[i]
#         single_item = f"""{item['skucode']}, {item['goodsname']}, {item['specification']}
# 标果价格: ¥{item['finalstandardprice']}
# 单价(斤,毛重): ¥{item['finalunitpricecatty']}
# 类目:{item['categoryname']}
# 毛重:{item['grossweight']}斤, 净重:{item['netweight']}斤
# 商品抽佣:{item.get('goodssiphoncommissionrate',0)}%, 向卖家抽佣:{item.get('sellersiphoncommissionrate',0)}%
# 月销量:{item.get('monthsale',0)}, 7日售后:{item.get('sevendayaftersale',0)}"""
#         csv_object[f"标果SKU{i+1}"] = single_item
#     sku_list_for_csv.append(csv_object)

# sku_list_for_csv_df = pd.DataFrame(sku_list_for_csv)
# sku_list_for_csv_df.to_csv(
#     f"./data/pop/顺鹿达SKU_vs_标果SKU_top20_{date_of_now}.csv", index=False
# )
# sku_list_for_csv_df.head(2)


# ## 使用语言模型（GPT3）做匹配

# In[ ]:


model = "gpt-35-turbo-16k"
# model = "gpt-4o"

llm_top_n = 3


def get_top_matched_biaoguo_sku(xianmu_sku_specification: str, biaoguo_sku_to_match=[]):
    sku_to_match = "\n".join(biaoguo_sku_to_match)
    messages = [
        {
            "role": "user",
            "content": f"以下商品中，和这个商品：“{xianmu_sku_specification}”最相似的前{llm_top_n}个是？请从商品名字、规格、包装净重这三个方面综合考虑。**请你直接返回你认为最相似的{llm_top_n}个商品，每行一个商品，不要返回其他任何信息**：\n{sku_to_match}",
        },
    ]
    completion = client.chat.completions.create(
        model=model, temperature=0.7, max_tokens=4095, messages=messages
    )

    response = completion.choices[0].message.content
    if (
        len(completion.choices) <= 0
        or f"{completion.choices[0].finish_reason}" == "content_filter"
    ):
        print(f"azure过滤了本次请求:{completion.choices[0].to_dict()}")
    if response is None:
        print(f"azure API返回了异常:{completion.to_dict()}")

    print(f"xianmu_title:{xianmu_sku_specification}\nresponse:{response}")
    return response


# get_top_matched_biaoguo_sku(xianmu_sku_specification=specification_test, biaoguo_sku_to_match=biaoguo_sku_to_match)


# ### 调用LLM进行匹配
# 
# - 太费事了，先这样吧

# In[ ]:


# def get_top_matched_by_llm(xianmu_sku):
#     specification = f"skucode:{xianmu_sku['sku']}, 品名:{xianmu_sku['categoryName']}-{xianmu_sku['sku_name']}"
#     specification = f"{specification}, 规格:{xianmu_sku['weight']}, 毛重:{xianmu_sku.get('weightNum', 0)*2}斤, 净重:{xianmu_sku.get('netWeightNum', 0)*2}斤"
#     top_matches = xianmu_sku["top_matches"]
#     biaoguo_sku_to_match = []
#     for biaoguo_matched in top_matches:
#         sku = f"skucode:{biaoguo_matched['skucode']}, 品名:{biaoguo_matched['categoryname']}-{biaoguo_matched['goodsname']}"
#         sku = f"{sku}, 规格:{biaoguo_matched['specification']}, 毛重:{biaoguo_matched['grossweight']}斤, 净重:{biaoguo_matched['netweight']}斤"
#         print(sku)
#         biaoguo_sku_to_match.append(sku)
#     return get_top_matched_biaoguo_sku(
#         xianmu_sku_specification=specification,
#         biaoguo_sku_to_match=biaoguo_sku_to_match,
#     )

# LLM太费事了，先这样吧
# sku_list_df["top_matched_gpt35"] = sku_list_df.apply(get_top_matched_by_llm, axis=1)


# ## 把鲜沐的SKU写入到本地CSV

# In[ ]:


sku_list_clean_df = sku_list_df[
    [
        "sku",
        "pdId",
        "categoryName",
        "openSale",
        "sku_name",
        "pd_name",
        "price",
        "supplyPrice",
        "areaName",
        "netWeightNum",
        "weightNum",
        "weight",
        "sku_spec",
        "spu_spec",
        "buyer_name",
        "picture",
    ]
].copy()
sku_list_clean_df.rename(
    columns={
        "categoryName": "类目",
        "openSale": "是否上架",
        "areaName": "运营区域",
        "netWeightNum": "净重(公斤)",
        "weightNum": "毛重(公斤)",
        "picture": "图片链接",
        "weight": "SKU规格",
        "price": "售价",
        "supplyPrice": "供应商提报价",
        "buyer_name": "买手名字",
    },
    inplace=True,
)
sku_list_clean_df["供应商提报价"].fillna(0, inplace=True)
sku_list_clean_df.sort_values(by=["类目", "pd_name"], inplace=True)
sku_list_clean_df.to_csv(f"./data/pop/鲜果POP全部SKU_{date_of_now}.csv", index=False)
sku_list_clean_df.head(10)


# ## 按照标果的销量排序

# In[ ]:


# all_sorted_sku = []
# added_sku_set = set()
# top_n_to_sort = 2

# for index, row in biaoguo_df.head(200)[
#     [
#         "top_matches",
#         "goodsname",
#         "categoryname",
#         "monthsale_gmv",
#         "monthsale",
#         "finalstandardprice",
#         "skucode",
#         "specification",
#         "grossweight",
#         "url",
#     ]
# ].iterrows():
#     new_row = {}
#     new_row["biaoguo_goodsname"] = row["goodsname"]
#     new_row["biaoguo_categoryname"] = row["categoryname"]
#     new_row["biaoguo_monthsale_gmv"] = row["monthsale_gmv"]
#     new_row["biaoguo_monthsale"] = row["monthsale"]
#     new_row["biaoguo_finalstandardprice"] = row["finalstandardprice"]
#     new_row["biaoguo_skucode"] = row["skucode"]
#     new_row["biaoguo_specification"] = row["specification"]
#     new_row["biaoguo_grossweight"] = row["grossweight"]
#     new_row["biaoguo_url"] = row["url"]
#     my_top_n = 0

#     for index, matched_item in enumerate(row["top_matches"]):
#         sku = matched_item["sku"]
#         if sku in added_sku_set:
#             print(f"duplicated sku:{sku}")
#             continue
#         on_sale = matched_item.get("onSale", "下架")
#         if "下架" in on_sale:
#             print(
#                 f"sku未上架:{sku},{matched_item.get('onSale',False)}, {matched_item['sku_name']}, price:{matched_item['price']}"
#             )
#             continue
#         my_top_n = my_top_n + 1
#         if my_top_n > top_n_to_sort:
#             print(f'为了防止太多重复的品...{matched_item["sku_name"]}')
#             break
#         new_row_with_xianmu = {}
#         new_row_with_xianmu.update(new_row)
#         new_row_with_xianmu["xianmu_sku"] = matched_item["sku"]
#         new_row_with_xianmu["xianmu_picture"] = matched_item["picture"]
#         new_row_with_xianmu["xianmu_price"] = matched_item["price"]
#         new_row_with_xianmu["xianmu_onSale"] = matched_item["onSale"]
#         new_row_with_xianmu["xianmu_weight"] = matched_item["weight"]
#         new_row_with_xianmu["xianmu_weightNum"] = matched_item["weightNum"]
#         new_row_with_xianmu["xianmu_sku_name"] = matched_item["sku_name"]
#         new_row_with_xianmu["xianmu_categoryName"] = matched_item["categoryName"]
#         new_row_with_xianmu["xianmu_buyer_name"] = matched_item["buyer_name"]
#         all_sorted_sku.append(new_row_with_xianmu)
#         added_sku_set.add(sku)

# all_sorted_sku_df = pd.DataFrame(all_sorted_sku)
# all_sorted_sku_df.head(10)


# In[ ]:


# sorted_html_content=[]
# for index, row in all_sorted_sku_df.iterrows():
#     content = f"""<div class="xianmu-container no-list-style">
#     <img class="large-img" src="{row['xianmu_picture']}">
#     <ul>
#     <li class="sku-title">{row['xianmu_sku']}, {row["xianmu_sku_name"]}</li>
#     <li>鲜沐售价: <span class="sku-price">¥{row["xianmu_price"]}, ¥{round(float(row["xianmu_price"])/float(row['xianmu_weightNum'])/2, 2)}/斤</span>(毛重)</li>
#     <li>规格:{row["xianmu_weight"]}, 毛重:{row.get('xianmu_weightNum', 0)*2}斤</li>
#     <li>买手:{row["xianmu_buyer_name"]}, 是否上架: {row["xianmu_onSale"]}</li>
#     <li class="sku-title">标果SKU:{row["biaoguo_skucode"]},{row["biaoguo_goodsname"]},{row["biaoguo_specification"]}</li>
#     <li>标果售价: <span class="sku-price">¥{row["biaoguo_finalstandardprice"]}, ¥{round(float(row["biaoguo_finalstandardprice"])/float(row['biaoguo_grossweight']),2)}/斤</span>(毛重)</li>
#     <li>标果月GMV: ¥{round(row["biaoguo_monthsale_gmv"],1)}, 月销{row["biaoguo_monthsale"]}件</li>
#     </ul></div>""".replace('\n', '')
#     sorted_html_content.append(content)

# html_content = f"""<html><head><meta charset="UTF-8"><meta name="title" content="POP商品首页排序-{date_of_now}">
# <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
# {css}
# <style>
# .xianmu-container{{font-size:smaller;}}
# .large-img{{object-fit: contain;width:160;}}
# .xianmu-container{{max-width:200px;padding:0.2vh 0.5vw;}}
# .sorted-container{{display: flex;width: 100%;flex-wrap: wrap;padding: 20px;}}
# .xianmu-container span.sku-price{{color:lightcoral;font-weight:bolder;}}
# </style>
# </head><body><div class="sorted-container">{''.join(sorted_html_content)}</div></body></html>"""

# file_path = f"./data/pop/POP商品首页排序-根据标果销量-{date_of_now}.html"

# # 保存HTML到本地文件：
# with open(file_path, "w", encoding="utf-8") as f:
#     f.write(html_content)

# print(f"写入HTML成功！{file_path}")


# ## 联合匹配标果、蜂果供、果速送、壹生鲜果等多家平台

# ### 标准SQL模型

# In[ ]:


from odps_client import get_odps_sql_result_as_df

ds_now = datetime.now().strftime("%Y%m%d")
ds_3days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")


def get_standard_df_by_sql_template(sql_template: str) -> pd.DataFrame:
    competitor_df = get_odps_sql_result_as_df(sql=sql_template.format(ds=ds_now))
    competitor_3d_df = get_odps_sql_result_as_df(
        sql=sql_template.format(ds=ds_3days_ago)
    )
    return merge_now_and_3d_data(
        competitor_df=competitor_df, competitor_3d_df=competitor_3d_df
    )


def merge_now_and_3d_data(
    competitor_df: pd.DataFrame, competitor_3d_df: pd.DataFrame
) -> pd.DataFrame:
    competitor_3d_df.rename(columns={"month_sale": "sales_volume_3d"}, inplace=True)
    competitor_df = pd.merge(
        competitor_df,
        competitor_3d_df[["sku_code", "sales_volume_3d"]],
        on="sku_code",
        how="left",
    )

    # Convert to numeric and handle errors
    competitor_df["month_sale"] = pd.to_numeric(
        competitor_df["month_sale"], errors="coerce"
    )
    competitor_df["sales_volume_3d"] = pd.to_numeric(
        competitor_df["sales_volume_3d"], errors="coerce"
    )

    # Fill NaNs with a default value (e.g., 0) or handle them as needed
    competitor_df["month_sale"].fillna(0, inplace=True)
    competitor_df["sales_volume_3d"].fillna(0, inplace=True)

    # Convert to integers
    competitor_df["month_sale"] = competitor_df["month_sale"].astype(int)
    competitor_df["sales_volume_3d"] = competitor_df["sales_volume_3d"].astype(int)

    # Perform the subtraction
    competitor_df["sales_volume_3d"] = (
        competitor_df["month_sale"] - competitor_df["sales_volume_3d"]
    )

    competitor_df.sort_values(by="sales_volume_3d", ascending=False, inplace=True)
    return competitor_df


# ### 蜂果供

# In[ ]:


fengguogong_sql = """
SELECT  categoryname AS category_name
        ,competitor
        ,goodscode AS sku_code
        ,spider_fetch_time
        ,NULL AS goods_siphon_commission_rate
        ,wholeggguoprice AS standard_price
        ,wholeggguoprice AS final_standard_price
        ,salenum AS month_sale
        ,goodslogo AS url
        ,NULL AS seller_siphon_commission_rate
        ,ggguoprice AS unit_price_catty
        ,priceunit AS unit
        ,NULL AS seller_name
        ,netweight AS gross_weight
        ,roughweight AS net_weight
        ,concat(sizedesc,pricedesc,',单果:',singlenutweight) AS specification
        ,goodsname AS goods_name
        ,NULL as seven_day_after_sale
FROM    (
            SELECT  *
                    ,RANK() OVER (PARTITION BY id,goodscode ORDER BY spider_fetch_time DESC ) AS rnk
            FROM    summerfarm_ds.spider_fengguogong_product_result_df
            WHERE   ds = '{ds}'
        ) 
WHERE   rnk = 1
;
"""

fengguogong_df = get_standard_df_by_sql_template(fengguogong_sql)
fengguogong_df.head(10)


# ### 标果-杭州

# In[ ]:


biaoguo_sql="""
SELECT  categoryname AS category_name,
        competitor,
        skucode AS sku_code,
        spider_fetch_time,
        goodssiphoncommissionrate AS goods_siphon_commission_rate,
        standardprice AS standard_price,
        finalstandardprice AS final_standard_price,
        monthsale AS month_sale,
        attachurlr AS url,
        sellersiphoncommissionrate AS seller_siphon_commission_rate,
        unitpricecatty AS unit_price_catty,
        unit,
        sellername AS seller_name,
        grossweight AS gross_weight,
        netweight AS net_weight,
        specification,
        goodsname AS goods_name,
        sevendayaftersale AS seven_day_after_sale
FROM(
            SELECT  *
                    ,RANK() OVER (PARTITION BY id,skucode ORDER BY spider_fetch_time DESC ) AS rnk
            FROM    summerfarm_ds.spider_biaoguo_with_prop_product_result_df
            WHERE   ds = '{ds}'
            AND     competitor = '标果-杭州' 
            AND     categoryname like '%/%'
        ) 
WHERE   rnk = 1
LIMIT   100000;
"""

biaoguo_df = get_standard_df_by_sql_template(biaoguo_sql)
biaoguo_df.head(10)


# ### 果速送

# In[ ]:


guosusong_sql = """
select   typename AS category_name,
    competitor,
    goodscode AS sku_code,
    spider_fetch_time,
    NULL AS goods_siphon_commission_rate,
    wholegssprice AS standard_price,
    wholegssprice AS final_standard_price,
    salenum AS month_sale,
    goodslogo AS url,
    NULL AS seller_siphon_commission_rate,
    roughprice AS unit_price_catty,
    priceunit AS unit,
    NULL AS seller_name,
    roughweight AS gross_weight,
    netweight AS net_weight,
    concat(sizedesc,',',pricedesc) AS specification,
    goodsname AS goods_name,
    NULL AS seven_day_after_sale
from (
            SELECT  *
                    ,RANK() OVER (PARTITION BY id,goodscode ORDER BY spider_fetch_time DESC ) AS rnk
            FROM    summerfarm_ds.spider_guosusong_product_result_df
            WHERE   ds = '{ds}'
        ) 
WHERE   rnk = 1
"""

guosusong_df = get_standard_df_by_sql_template(guosusong_sql)
guosusong_df.head(10)


# ### 壹生鲜果

# In[ ]:


yishengxianguo_sql = """
select * from (
    select category_name AS category_name,
        competitor AS competitor,
        goods_code AS sku_code,
        spider_fetch_time AS spider_fetch_time,
        sales_volume AS month_sale,
        pic_path AS url,
        min_unit AS unit,
        provider_name AS seller_name,
        goods_describe AS specification,
        goods_name AS goods_name,
        item_vos,basics_item,RANK() OVER (PARTITION BY goods_code ORDER BY spider_fetch_time DESC ) AS rnk
    from summerfarm_ds.spider_yishengxianguo_product_result_df 
    where ds='{ds}'
)t
where rnk=1
"""

yishengxianguo_df = get_odps_sql_result_as_df(sql=yishengxianguo_sql.format(ds=ds_now))
yishengxianguo_3d_df = get_odps_sql_result_as_df(
    sql=yishengxianguo_sql.format(ds=ds_3days_ago)
)

import re


def extract_gross_weight(item_name):
    # Use regex to find the number before '斤'
    match = re.search(r"(\d+)(?=斤)", item_name)
    if match:
        return int(match.group(1))
    else:
        return None


def convert_yishengxianguo_df(yishengxianguo_df: pd.DataFrame) -> pd.DataFrame:
    all_yishengxianguo_items = []
    for inx, row in yishengxianguo_df.iterrows():
        item = {}
        item["category_name"] = row["category_name"]
        item["competitor"] = row["competitor"]
        item["sku_code"] = row["sku_code"]
        item["spider_fetch_time"] = row["spider_fetch_time"]
        item["url"] = row["url"]
        item["unit"] = row["unit"]
        item["seller_name"] = row["seller_name"]
        item["specification"] = row["specification"]
        item["goods_name"] = row["goods_name"]

        json_string = (
            row["item_vos"]
            .replace("'", '"')
            .replace("None", "null")
            .replace("False", "false")
            .replace("True", "true")
        )
        # print(json_string)
        # Convert single quotes to double quotes
        item_vo = json.loads(json_string)[0]
        # print(item_vo)
        item["goods_name"] = row["goods_name"]
        item["specification"] = f"{item['specification']}, {item_vo['itemName']}"
        item["final_standard_price"] = item_vo["price"]
        item["standard_price"] = item_vo["price"]
        item["seller_siphon_commission_rate"] = ""
        item["goods_siphon_commission_rate"] = ""
        item["month_sale"] = item_vo["saleNum"]
        item["gross_weight"] = extract_gross_weight(item_vo["itemName"])
        item["unit_price_catty"] = (
            float(item_vo["price"]) / item["gross_weight"]
            if item["gross_weight"] is not None
            else 0
        )
        item["net_weight"] = "-1"
        item["seven_day_after_sale"] = None
        all_yishengxianguo_items.append(item)
    return pd.DataFrame(all_yishengxianguo_items)


yishengxianguo_df = convert_yishengxianguo_df(yishengxianguo_df)
yishengxianguo_3d_df = convert_yishengxianguo_df(yishengxianguo_3d_df)

yishengxianguo_df = merge_now_and_3d_data(
    competitor_df=yishengxianguo_df, competitor_3d_df=yishengxianguo_3d_df
)
yishengxianguo_df.head(10)


# ### 合并起来

# In[ ]:


# Concatenating DataFrames
all_competitor_df = pd.concat([biaoguo_df, guosusong_df, fengguogong_df, yishengxianguo_df], ignore_index=True)

categories=all_competitor_df['category_name'].unique()

print(f"所有的类目:{categories}")

# List of known fruit categories
fruit_keywords = ['榴莲', '火龙果', '菠萝', '山竹', '菠萝蜜', '蓝莓', '梨', '椰子', '柠檬', '猕猴桃', '牛油果', '芒果', '苹果', '柚',
                  '葡萄', '桃', '李', '西瓜', '枣', '樱桃', '草莓', '橙', '荔枝', '柿子', '石榴', '杏', '桔', '柑', '甜瓜', '提子', '木瓜',
                  '杨桃', '人参果', '释迦果', '莲雾', '芭乐', '百香果', '黄瓜']

# Filter categories based on the fruit keywords
fruit_categories = [category for category in categories if any(keyword in category for keyword in fruit_keywords)]

# Filter rows where category_name is in fruit_categories
fruit_filtered_df = all_competitor_df[all_competitor_df['category_name'].isin(fruit_categories)]

# Filter rows where category_name is NOT in fruit_categories
not_fruit_filtered_df = all_competitor_df[~all_competitor_df['category_name'].isin(fruit_categories)]

print(f"过滤后的数量:{len(fruit_filtered_df)}, 过滤前的数量:{len(all_competitor_df)}")



fruit_filtered_df["sku_embeddings"] = fruit_filtered_df.apply(get_competitor_like_biaoguo_embedding, axis=1)
sku_list_df["sku_embeddings"] = sku_list_df.apply(get_xianmu_embedding, axis=1)

fruit_filtered_df["category_embeddings"] = fruit_filtered_df['category_name'].apply(get_category_embedding)
sku_list_df["category_embeddings"] = sku_list_df['categoryName'].apply(get_category_embedding)


# In[ ]:


fruit_filtered_df["monthsale_gmv"] = fruit_filtered_df["month_sale"].astype(
    float
) * fruit_filtered_df["final_standard_price"].astype(float)

fruit_filtered_df["last_3d_gmv"] = fruit_filtered_df[
    "sales_volume_3d"
] * fruit_filtered_df["final_standard_price"].astype(float)

fruit_filtered_df.sort_values(
    by=["competitor", "last_3d_gmv", "monthsale_gmv"], ascending=False, inplace=True
)
fruit_filtered_df.head(20)[
    [
        "last_3d_gmv",
        "monthsale_gmv",
        "sales_volume_3d",
        "month_sale",
        "goods_name",
        "competitor",
    ]
]

all_competitor_to_xianmu_top_n = 10


## 把表格的商品映射到鲜沐上，标果的商品作为主键，和 def find_top_matches_for_xianmu() 的顺序是反过来的
def find_top_xianmu_matches_for_competitor(competitor_sku_row: pd.Series):
    similarities = []
    for index, xianmu_row in sku_list_df.iterrows():
        category_similarity_score = calculate_cosine_similarity(
            competitor_sku_row["category_embeddings"], xianmu_row["category_embeddings"]
        )
        if category_similarity_score <= 0.88:
            # 如果类目的相似度小于0.88，则不需要再考虑名字的embedding
            print(
                f'类目的相似度小于0.88:{category_similarity_score}, {competitor_sku_row["competitor"]}类目:{competitor_sku_row["category_name"]}, 鲜沐类目:{xianmu_row["categoryName"]}'
            )
            continue
        similarity_score = calculate_cosine_similarity(
            competitor_sku_row["sku_embeddings"], xianmu_row["sku_embeddings"]
        )
        # 类目embedding + sku embedding分数
        similarities.append(
            (xianmu_row.to_dict(), similarity_score + category_similarity_score)
        )

    # Sort the results based on similarity score in descending order
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Get the top N matches
    top_n_matches = [
        item[0] for item in sorted_similarities[:all_competitor_to_xianmu_top_n]
    ]
    print(
        f"Matched rows for:{competitor_sku_row['goods_name']}, {[xianmu.get('sku_name') for xianmu in top_n_matches]}"
    )
    competitor_sku_row["top_matches"] = top_n_matches
    return top_n_matches


fruit_filtered_df["top_matches"] = fruit_filtered_df.apply(
    find_top_xianmu_matches_for_competitor, axis=1
)
fruit_filtered_df.head(10)[
    [
        "top_matches",
        "goods_name",
        "category_name",
        "competitor",
        "last_3d_gmv",
        "monthsale_gmv",
    ]
]


# In[ ]:


import pandasql

fruit_filtered_df["category_level2"] = (
    fruit_filtered_df["category_name"].str.split("/").str[0]
)

fruit_filtered_clean_df = fruit_filtered_df[
    [
        "sku_code",
        "goods_name",
        "category_name",
        "month_sale",
        "monthsale_gmv",
        "last_3d_gmv",
        "category_level2",
        "competitor",
    ]
]

static_df = pandasql.sqldf(
    """select category_name,competitor
    ,sum(monthsale_gmv) as category_monthsale_gmv 
    ,sum(last_3d_gmv) as category_3d_gmv 
    from fruit_filtered_clean_df 
    group by competitor,category_name"""
)

# Step 2: Merge total_gmv into fruit_filtered_df
fruit_filtered_with_category_gmv_df = fruit_filtered_df.merge(
    static_df[
        ["category_name", "competitor", "category_monthsale_gmv", "category_3d_gmv"]
    ],
    on=["category_name", "competitor"],
    how="left",
)

fruit_filtered_with_category_gmv_df.head(10)

fruit_filtered_with_category_gmv_df["monthsale_gmv_percentile_of_category"] = round(
    100.00
    * fruit_filtered_with_category_gmv_df["monthsale_gmv"].astype(float)
    / fruit_filtered_with_category_gmv_df["category_monthsale_gmv"].astype(float),
    2,
)

fruit_filtered_with_category_gmv_df.head(10)

fruit_filtered_with_category_gmv_df["last_3d_gmv_percentile_of_category"] = round(
    100.00
    * fruit_filtered_with_category_gmv_df["last_3d_gmv"].astype(float)
    / fruit_filtered_with_category_gmv_df["category_3d_gmv"].astype(float),
    2,
)


# Step 4: Rank each item within its category based on monthsale_gmv
fruit_filtered_with_category_gmv_df["category_rank"] = (
    fruit_filtered_with_category_gmv_df.groupby(["category_name", "competitor"])[
        "last_3d_gmv"
    ].rank(ascending=False)
)

fruit_filtered_top10_of_each_category_df = fruit_filtered_with_category_gmv_df[
    fruit_filtered_with_category_gmv_df["category_rank"] <= 10.0
]
fruit_filtered_top10_of_each_category_df = fruit_filtered_top10_of_each_category_df[
    fruit_filtered_top10_of_each_category_df["category_monthsale_gmv"] > 1000.00
]

fruit_filtered_top10_of_each_category_df.head(10)[
    [
        "category_monthsale_gmv",
        "last_3d_gmv",
        "monthsale_gmv",
        "last_3d_gmv_percentile_of_category",
        "monthsale_gmv_percentile_of_category",
        "category_rank",
        "category_level2",
        "category_name",
        "goods_name",
        "competitor",
    ]
]


# In[ ]:


from datetime import datetime, timedelta

date_of_now = datetime.now().strftime("%Y-%m-%d")
sku_list_for_csv_top200 = []
for _, row in fruit_filtered_top10_of_each_category_df.iterrows():
    competitor = row["competitor"]
    csv_object = {"competitor": competitor}
    unit_price_catty = row["unit_price_catty"]
    gross_weight = row.get("gross_weight", "")
    if "" == gross_weight:
        print(f"未找到gross_weight:{gross_weight}")
    else:
        unit_price_catty = round(
            float(row.get("final_standard_price", "0.0"))
            / float(row.get("gross_weight", "1")),
            2,
        )
    competitor_content = f"""{row["sku_code"]},{row["goods_name"]},{row["specification"]}
{competitor}售价: ¥{row.get("final_standard_price","0.0")}, ¥{unit_price_catty}/斤(毛重)
{competitor}月GMV: ¥{round(row["monthsale_gmv"],1)}, 月销{row["month_sale"]}件
{competitor}近3日GMV: ¥{round(row["last_3d_gmv"],1)}, 近3日销{row["sales_volume_3d"]}件
毛重: {row["gross_weight"]}斤, 商品抽佣:{row.get('goods_siphon_commission_rate',0)}%, 向卖家抽佣:{row.get('seller_siphon_commission_rate',0)}%"""
    csv_object["竞争对手skucode"] = row["sku_code"]
    csv_object["竞争对手类目"] = f"""{row["category_name"]}"""
    csv_object["类目月GMV"] = f"{row['category_monthsale_gmv']}"
    csv_object["类目近3日GMV"] = f"{row['category_3d_gmv']}"
    csv_object["近3日占类目GMV百分比"] = f"{row['last_3d_gmv_percentile_of_category']}%"
    csv_object["月GMV占类目百分比"] = f"{row['monthsale_gmv_percentile_of_category']}%"
    csv_object["竞争对手SKU售价"] = (
        f"""¥{row.get("final_standard_price","0.0")}, ¥{unit_price_catty}/斤(毛重)"""
    )
    csv_object["竞争对手SKU月GMV"] = (
        f"""¥{round(row["monthsale_gmv"],1)}, 月销{row["month_sale"]}件"""
    )
    csv_object["竞争对手SKU近3日GMV"] = (
        f"""¥{round(row["last_3d_gmv"],1)}, 近3日销{row["sales_volume_3d"]}件"""
    )
    csv_object["竞争对手SKU"] = competitor_content

    top_matches = row["top_matches"]
    for i in range(0, 20):
        if i >= len(top_matches):
            break
        item = top_matches[i]
        single_item = f"""{item['sku']}, {item['sku_name']}, {item['weight']}
顺鹿达价格: ¥{item['price']}
单价: ¥{round(float(item['price'])/float(item['weightNum'])/2,2)}/斤(毛重)
类目:{item['categoryName']}, 毛重:{item['weightNum']*2}斤"""
        csv_object[f"顺鹿达SKU{i+1}"] = single_item
    sku_list_for_csv_top200.append(csv_object)

sku_list_for_csv_top200_df = pd.DataFrame(sku_list_for_csv_top200)
sku_list_for_csv_top200_df.drop_duplicates(
    subset=[f"竞争对手skucode", "competitor"], inplace=True
)
sku_list_for_csv_top200_df.to_csv(
    f"./data/pop/顺鹿达SKU_vs_竞争对手SKU_top热销200_{date_of_now}.csv",
    index=False,
)

for competitor in sku_list_for_csv_top200_df["competitor"].unique():
    print(competitor)
    competitor_df = sku_list_for_csv_top200_df[
        sku_list_for_csv_top200_df["competitor"] == competitor
    ]
    competitor_df.to_csv(
        f"./data/pop/顺鹿达SKU_vs_{competitor}SKU_top热销200_{date_of_now}.csv",
        index=False,
    )


# ### 展示一下HTML

# In[ ]:


# sorted_html_content=[]
# for index, row in fruit_filtered_df.iterrows():
#     content = f"""<div class="xianmu-container no-list-style">
#     <img class="large-img" src="{row['xianmu_picture']}">
#     <ul>
#     <li class="sku-title">{row['xianmu_sku']}, {row["xianmu_sku_name"]}</li>
#     <li>鲜沐售价: <span class="sku-price">¥{row["xianmu_price"]}, ¥{round(float(row["xianmu_price"])/float(row['xianmu_weightNum'])/2, 2)}/斤</span>(毛重)</li>
#     <li>规格:{row["xianmu_weight"]}, 毛重:{row.get('xianmu_weightNum', 0)*2}斤</li>
#     <li>买手:{row["xianmu_buyer_name"]}, 是否上架: {row["xianmu_onSale"]}</li>
#     <li class="sku-title">标果SKU:{row["biaoguo_skucode"]},{row["biaoguo_goodsname"]},{row["biaoguo_specification"]}</li>
#     <li>标果售价: <span class="sku-price">¥{row["biaoguo_finalstandardprice"]}, ¥{round(float(row["biaoguo_finalstandardprice"])/float(row['biaoguo_grossweight']),2)}/斤</span>(毛重)</li>
#     <li>标果月GMV: ¥{round(row["biaoguo_monthsale_gmv"],1)}, 月销{row["biaoguo_monthsale"]}件</li>
#     </ul></div>""".replace('\n', '')
#     sorted_html_content.append(content)

# html_content = f"""<html><head><meta charset="UTF-8"><meta name="title" content="POP商品首页排序-{date_of_now}">
# <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
# {css}
# <style>
# .xianmu-container{{font-size:smaller;}}
# .large-img{{object-fit: contain;width:160;}}
# .xianmu-container{{max-width:200px;padding:0.2vh 0.5vw;}}
# .sorted-container{{display: flex;width: 100%;flex-wrap: wrap;padding: 20px;}}
# .xianmu-container span.sku-price{{color:lightcoral;font-weight:bolder;}}
# </style>
# </head><body><div class="sorted-container">{''.join(sorted_html_content)}</div></body></html>"""

# file_path = f"./data/pop/POP商品首页排序-根据标果销量-{date_of_now}.html"

# # 保存HTML到本地文件：
# with open(file_path, "w", encoding="utf-8") as f:
#     f.write(html_content)

# print(f"写入HTML成功！{file_path}")

