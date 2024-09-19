import random,time,requests
from odps import ODPS
from odps import options
import math

def generate_random_number(length=8):
  """Generate a random number with a specified length."""
  return ''.join([str(random.randint(0, 9)) for _ in range(length)])

def login_summerfarm(is_dev,user_id,psd):
    online_url = "https://admin.summerfarm.net/authentication/auth/username/login"
    dev_url = "https://devadmin.summerfarm.net/authentication/auth/username/login"

    if is_dev:
        url = dev_url
    else:
        url = online_url

    # 定义Headers
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://admin.summerfarm.net",
        "Referer": "https://admin.summerfarm.net/summerfarm/home.html",
        "Sec-Ch-Ua": '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Xm-Biz": "xm-manage",
        "Xm-Phone": "",
        "Xm-Platform": "web",
        "Xm-Rqid": str(int(time.time()*1000))+'-'+str(generate_random_number()),
        "Xm-Uid": ""
    }

    # 定义Payload
    data = {
        "username": user_id,
        "password": psd
    }

    # 发起POST请求
    response = requests.post(url, headers=headers, data=data)
    # print(response.text)
    return response.json()['data']['token']

def change_sku_price(is_dev,login_token,spu,disc,price,sku_id,area_list):

    online_url = "https://admin.summerfarm.net/price-adjust/saveAdjustSheet"
    dev_url = "https://devadmin.summerfarm.net/price-adjust/saveAdjustSheet"

    if is_dev:
        url = dev_url
    else:
        url = online_url

    data = [{
        "pdName": spu,
        "sku": sku_id,
        "weight": disc,
        "deleteLadderPrice": "true",
        "areaNoList": area_list,
        "price": float(price)
    }]

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/json;charset=UTF-8",
        "Cookie": "Hm_lvt_4c79036bfe07bfe7af679f50c1adaf0b=1701775138",
        "Origin": "https://admin.summerfarm.net",
        "Priority": "u=1, i",
        "Referer": "https://admin.summerfarm.net/summerfarm/home.html",
        "Sec-Ch-Ua": "\"Chromium\";v=\"124\", \"Google Chrome\";v=\"124\", \"Not-A.Brand\";v=\"99\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Token":  login_token,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Xm-Biz": "xm-manage",
        # "Xm-Phone": "18320949762",
        "Xm-Platform": "web",
        "Xm-Rqid": str(int(time.time()*1000))+'-'+str(generate_random_number()),
        # "Xm-Uid": "10488"
    }

    # 发送POST请求
    response = requests.post(url, json=data, headers=headers)
    return response.json()


def get_base_price_model_sql():
    headers = ['spu_name', 'new_price', 'sku_id', 'disc','city_id_list','warehouse_name','large_area_name']

    sql_code = f"""
SELECT spu_name,new_price,sku_id,disc,city_id_list,warehouse_name,large_area_name
FROM 
(
    SELECT *
        ,row_number() over(partition by warehouse_name,large_area_name,sku_id order by new_price) as rank
    FROM 
    (
        SELECT a4.spu_name
            ,round(unit_cost/(1-a1.aim_profit),1) as new_price
            ,a4.sku_id,a4.disc
            ,a5.city_id_list
            ,a1.warehouse_name
            ,a1.large_area_name
        FROM 
        (
            select *
            FROM summerfarm_ds_dev.tmp_xunzhi_price_v1
            where stg_type = "spu"
        )a1
        JOIN 
        (
            select DISTINCT ds,sku,warehouse_no,spu,warehouse_name
            from summerfarm_tech.dws_prd_on_out_sale_di
            where ds = MAX_PT("summerfarm_tech.dws_prd_on_out_sale_di")
            and duration_on_shelf > duration_sale_out
        )a2
        on a1.warehouse_name=a2.warehouse_name and a1.sku_id=a2.sku
        LEFT JOIN 
        (
            SELECT sku,warehouse_name,area_no
                ,round(sum(last_store_quantity*cost)/sum(last_store_quantity),2) as unit_cost
            from summerfarm_ds.dwd_sp_chain_status_v2
            where ds = MAX_PT("summerfarm_ds.dwd_sp_chain_status_v2")
            GROUP BY sku,warehouse_name,area_no
            HAVING sum(last_store_quantity) > 0
        )a3
        on a1.sku_id=a3.sku and a1.warehouse_name=a3.warehouse_name
        LEFT JOIN 
        (
            SELECT distinct ds,category2,category3,category4,spu_id,spu_name,sku_id,sku_spec,disc,weight,volume,standard_cnt,standard_unit
            ,regexp_extract(disc, '(一级|二级|三级|普通|精品)', 1) AS grade_level
            ,create_date
            from summerfarm_tech.dim_sku_df
            where ds = MAX_PT("summerfarm_tech.dim_sku_df")
        )a4
        on a1.sku_id=a4.sku_id
        JOIN 
        (
            select large_area_name,COLLECT_LIST(city_id) as city_id_list
            from summerfarm_tech.dim_city_df
            where ds = MAX_PT("summerfarm_tech.dim_city_df")
            and LENGTH(city_name) <= 2
            GROUP BY large_area_name
        )a5
        on a1.large_area_name=a5.large_area_name
        WHERE a3.unit_cost IS not NULL

        UNION 

        SELECT a4.spu_name
            ,round(unit_cost/(1-a1.aim_profit),1) as new_price
            ,a4.sku_id,a4.disc
            ,a5.city_id_list
            ,a1.warehouse_name
            ,a1.large_area_name
        FROM 
        (
            select *
            FROM summerfarm_ds_dev.tmp_xunzhi_price_v1
            where stg_type = "spu"
        )a1
        JOIN 
        (
            select DISTINCT ds,sku,warehouse_no,spu,warehouse_name
            from summerfarm_tech.dws_prd_on_out_sale_di
            where ds = MAX_PT("summerfarm_tech.dws_prd_on_out_sale_di")
            and duration_on_shelf > duration_sale_out
        )a2
        on a1.warehouse_name=a2.warehouse_name and a1.spu_name=a2.spu
        LEFT JOIN 
        (
            SELECT sku,warehouse_name,area_no
                ,round(sum(last_store_quantity*cost)/sum(last_store_quantity),2) as unit_cost
            from summerfarm_ds.dwd_sp_chain_status_v2
            where ds = MAX_PT("summerfarm_ds.dwd_sp_chain_status_v2")
            GROUP BY sku,warehouse_name,area_no
            HAVING sum(last_store_quantity) > 0
        )a3
        on a2.sku=a3.sku and a2.warehouse_name=a3.warehouse_name
        LEFT JOIN 
        (
            SELECT distinct ds,category2,category3,category4,spu_id,spu_name,sku_id,sku_spec,disc,weight,volume,standard_cnt,standard_unit
            ,regexp_extract(disc, '(一级|二级|三级|普通|精品)', 1) AS grade_level
            ,create_date
            from summerfarm_tech.dim_sku_df
            where ds = MAX_PT("summerfarm_tech.dim_sku_df")
        )a4
        on a2.sku=a4.sku_id
        JOIN 
        (
            select large_area_name,COLLECT_LIST(city_id) as city_id_list
                ,COLLECT_LIST(city_name) as city_name
            from summerfarm_tech.dim_city_df
            where ds = MAX_PT("summerfarm_tech.dim_city_df")
            and LENGTH(city_name) <= 2
            GROUP BY large_area_name
        )a5
        on a1.large_area_name=a5.large_area_name
        WHERE a3.unit_cost IS not NULL
    )b1
)c1
    """

    return sql_code, headers

def get_by_sql(sql_code,headers,access_id,secret_access_key):
    # --设置order limit超过10000行也执行
    options.sql.settings = {
        'odps.sql.validate.orderby.limit': False,
        'odps.sql.submit.mode': 'script'
    }

    # 将表查询数据写入到当前文件中
    o = ODPS(access_id=access_id, secret_access_key=secret_access_key, project="summerfarm_ds", endpoint="http://service.cn-hangzhou.maxcompute.aliyun.com/api")
    data = []

    reader = o.execute_sql(sql_code).open_reader(tunnel=True, limit=False)
    for record in reader:
        row = []
        for word in headers:
            row.append(record[word])
        data.append(row)

    return data

def change_price_v1(user_id, psd,access_id, secret_access_key):
    # 业务逻辑，不要调整
    sql_code, headers = get_base_price_model_sql()
    data = get_by_sql(sql_code, headers, access_id, secret_access_key)
    is_dev = False
    login_token = login_summerfarm(is_dev, user_id, psd)

    for r in data:
        spu = r[0]
        new_price = math.ceil(float(r[1]))
        sku_id = r[2]
        disc = r[3]
        city_id_list = r[4]

        res = change_sku_price(is_dev, login_token, spu, disc, new_price, sku_id, city_id_list)
        print(res, sku_id)

#鲜沐商城账号密码
user_id = ""
psd = ""

#阿里云api-key
access_id = ""
secret_access_key = ""

#修改价格
change_price_v1(user_id, psd,access_id, secret_access_key)