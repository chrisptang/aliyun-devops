#!/usr/bin/env python
# coding: utf-8

"""
春节运费规则批量修改脚本

用法:
    # 在qaadmin环境增加运费（dry-run模式，只备份不修改）
    python update-delivery-fee-spring-festival-2026.py --env qaadmin --dry-run

    # 在admin环境增加运费（实际执行，默认10元）
    python update-delivery-fee-spring-festival-2026.py --env admin

    # 指定运费金额为15元
    python update-delivery-fee-spring-festival-2026.py --env admin --fee 15

    # 从备份文件恢复运费规则
    python update-delivery-fee-spring-festival-2026.py --env admin --restore --backup-date 20260130
"""

import argparse
import copy
import json
import os
from datetime import datetime

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="春节运费规则批量修改脚本")
    parser.add_argument(
        "--env",
        type=str,
        choices=["qaadmin", "admin"],
        default="qaadmin",
        help="目标环境: qaadmin 或 admin (默认: qaadmin)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="从备份文件恢复运费规则 (默认是增加运费)",
    )
    parser.add_argument(
        "--backup-date",
        type=str,
        default=None,
        help="备份文件日期，格式如 20260130 (restore操作必须指定)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只备份，不实际更新价格",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=10.0,
        help="0元运费规则的新价格 (默认: 10元)",
    )
    parser.add_argument(
        "--area",
        type=int,
        default=None,
        help="指定单个运营服务区编号，仅对该区域操作",
    )
    return parser.parse_args()


def get_token(env: str) -> tuple[str, dict]:
    """获取登录token"""
    login_url = f"https://{env}.summerfarm.net/authentication/auth/username/login"
    login_data = {
        "username": "peng.tang@summerfarm.net",
        "password": os.getenv(
            f"XIANMU_ADMIN_PASSWORD_{env}", os.getenv("XIANMU_ADMIN_PASSWORD")
        ),
    }

    token = requests.post(login_url, data=login_data).json()
    print(f"登录响应: {token}")

    token_str = token.get("data").get("token")
    headers = {
        "token": token_str,
        "xm-rqid": "批量修改春节运费",
        "xm-uid": "2047",
        "Content-Type": "application/json;charset=UTF-8",
    }
    return token_str, headers


def get_all_active_area_nos(env: str, token_str: str) -> list[int]:
    """
    从API获取所有启用状态的运营服务区编号
    遍历所有大区，获取status=true的区域
    """
    area_nos = []
    page = 1
    total_pages = None

    while total_pages is None or page <= total_pages:
        url = f"https://{env}.summerfarm.net/large-area/v2/{page}/100"
        headers = {
            "accept": "application/json, text/plain, */*",
            "token": token_str,
            "xm-rqid": "get_all_areas",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != 200:
                print(f"获取大区列表失败: {data}")
                break

            result = data.get("data", {})
            total_pages = result.get("pages", 1)

            for large_area in result.get("list", []):
                for area in large_area.get("areaList", []):
                    # 只获取status=true的区域
                    if area.get("status") is True:
                        area_nos.append(area.get("areaNo"))

            page += 1

        except Exception as e:
            print(f"获取大区列表出错: {e}")
            break

    print(f"共获取到 {len(area_nos)} 个启用状态的运营服务区")
    return area_nos


def get_area_delivery_fee_detail(
    area_no: int, token_str: str, env: str, from_local_cache: bool = False, cache_file: str = None
) -> dict:
    """获取区域配送费规则详情"""
    if from_local_cache:
        if not cache_file:
            raise ValueError("从本地缓存读取时必须指定cache_file")
        with open(cache_file, "r") as f:
            backup_json = json.load(f)
        return backup_json.get(str(area_no))

    data = {"type": 3, "businessId": area_no}
    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json;charset=UTF-8",
        "token": token_str,
        "xm-rqid": "chunjie_delivery_fee_10yuan",
    }
    rules = None
    try:
        response = requests.post(
            f"https://{env}.summerfarm.net/marketing-center/delivery-fee-rule/query/detail",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        rules = response.text
        print(rules)
        rules_json = json.loads(rules)
        if rules_json and "data" in rules_json:
            return rules_json.get("data")
        else:
            return {}
    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}, rules: {rules}")
        return {}
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError encountered: {e}, rules: {rules}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def transform_rules_for_increase(rules_of_area: dict, increase_amount: float = 10.0) -> tuple[dict, bool]:
    """
    转换规则用于increase操作：将0元运费增加指定金额
    返回: (transformed_rules, modified)
    """
    modified = False
    transformed_rules = {
        "type": rules_of_area.get("type"),
        "businessId": int(rules_of_area.get("businessId")),
        "ruleInputList": [],
    }

    for rule in rules_of_area.get("ruleVOList", []):
        rule_input = {
            "ageing": rule.get("ageing"),
            "startDeliveryAmount": rule.get("startDeliveryAmount"),
            "categoryRuleInputList": [],
        }

        for category_rule in rule.get("categoryRuleVOList", []):
            delivery_fee = category_rule.get("deliveryFee", 0)
            express_fee = category_rule.get("expressFee", 0)

            # 检查是否需要修改（0元运费加10元）
            new_delivery_fee = delivery_fee
            new_express_fee = express_fee

            if delivery_fee <= 0:
                new_delivery_fee = increase_amount
                modified = True
            if express_fee <= 0:
                new_express_fee = increase_amount
                modified = True

            category_rule_input = {
                "stepValue": float(category_rule.get("stepValue")),
                "deliveryFee": new_delivery_fee,
                "expressFee": new_express_fee,
                "feeMode": category_rule.get("feeMode"),
                "categoryType": category_rule.get("categoryType"),
            }
            rule_input["categoryRuleInputList"].append(category_rule_input)

        transformed_rules["ruleInputList"].append(rule_input)

    return transformed_rules, modified


def transform_rules_for_restore(rules_of_area: dict) -> dict:
    """
    转换规则用于restore操作：从备份恢复原始价格
    """
    transformed_rules = {
        "type": rules_of_area.get("type"),
        "businessId": int(rules_of_area.get("businessId")),
        "ruleInputList": [],
    }

    for rule in rules_of_area.get("ruleVOList", []):
        rule_input = {
            "ageing": rule.get("ageing"),
            "startDeliveryAmount": rule.get("startDeliveryAmount"),
            "categoryRuleInputList": [],
        }

        for category_rule in rule.get("categoryRuleVOList", []):
            category_rule_input = {
                "stepValue": float(category_rule.get("stepValue")),
                "deliveryFee": float(category_rule.get("deliveryFee", 0)),
                "expressFee": float(category_rule.get("expressFee", 0)),
                "feeMode": category_rule.get("feeMode"),
                "categoryType": category_rule.get("categoryType"),
            }
            rule_input["categoryRuleInputList"].append(category_rule_input)

        transformed_rules["ruleInputList"].append(rule_input)

    return transformed_rules


def update_delivery_fee_rules(env: str, token_str: str, objects_to_update: list, dry_run: bool = False):
    """发送请求更新运费规则"""
    if dry_run:
        print("\n[DRY-RUN 模式] 以下规则将被更新（实际未执行）:")
        for obj in objects_to_update:
            print(f"  区域 {obj['businessId']}: {len(obj['ruleInputList'])} 条规则")
        return

    if len(objects_to_update) > 0:
        for obj in objects_to_update:
            url = f"https://{env}.summerfarm.net/marketing-center/delivery-fee-rule/upsert/save"
            headers = {
                "accept": "application/json, text/plain, */*",
                "content-type": "application/json;charset=UTF-8",
                "token": token_str,
                "xm-phone": "18618107293",
                "xm-rqid": "chunjie_update_delivery_fee_10",
            }

            response = requests.post(url, headers=headers, json=obj)
            print(f"Response for obj: {obj}\n{response.status_code}, {response.text}")


def main():
    args = parse_args()

    # restore操作必须指定backup-date
    if args.restore and not args.backup_date:
        print("错误: --restore 操作必须指定 --backup-date 参数")
        return

    operation = "restore" if args.restore else "increase"
    print(f"环境: {args.env}")
    print(f"操作: {operation}")
    print(f"Dry-run: {args.dry_run}")
    if not args.restore:
        print(f"运费金额: {args.fee}元")
    if args.backup_date:
        print(f"备份日期: {args.backup_date}")
    if args.area:
        print(f"指定区域: {args.area}")

    # 获取token
    token_str, headers = get_token(args.env)
    print(f"Headers: {headers}")

    # 确定要处理的运营服务区列表
    if args.area:
        area_no_list_to_update = [args.area]
    else:
        # 从API获取所有启用状态的运营服务区
        area_no_list_to_update = get_all_active_area_nos(args.env, token_str)

    objects_to_update = []
    no_need_to_modify = []
    backup_rules = {}

    if not args.restore:
        # increase操作：从远程获取当前规则，给0元运费加指定金额
        for area_no in area_no_list_to_update:
            rules_of_area = get_area_delivery_fee_detail(area_no, token_str, args.env, from_local_cache=False)
            if not rules_of_area:
                print(f"获取配送规则错误: {area_no}")
                continue

            # 备份原始规则
            backup_rules[area_no] = copy.deepcopy(rules_of_area)

            # 转换并检查是否需要修改
            transformed_rules, modified = transform_rules_for_increase(rules_of_area, increase_amount=args.fee)

            if modified:
                objects_to_update.append(transformed_rules)
            else:
                print(f"这个区域不需要改规则: {area_no}")
                no_need_to_modify.append(rules_of_area)

        # 保存备份
        yearmonth = datetime.now().strftime("%Y%m%d")
        if args.area:
            backup_file = f"./delivery_fee_rules_backup_{yearmonth}-{args.env}-area{args.area}.json"
        else:
            backup_file = f"./delivery_fee_rules_backup_{yearmonth}-{args.env}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_rules, f, ensure_ascii=False, indent=2)
        print(f"原始规则已备份到 {backup_file}")

    else:
        # restore操作：从本地备份文件恢复
        if args.area:
            backup_file = f"./delivery_fee_rules_backup_{args.backup_date}-{args.env}-area{args.area}.json"
        else:
            backup_file = f"./delivery_fee_rules_backup_{args.backup_date}-{args.env}.json"
        if not os.path.exists(backup_file):
            print(f"错误: 备份文件不存在: {backup_file}")
            return

        print(f"从备份文件恢复: {backup_file}")

        for area_no in area_no_list_to_update:
            rules_of_area = get_area_delivery_fee_detail(
                area_no, token_str, args.env, from_local_cache=True, cache_file=backup_file
            )
            if not rules_of_area:
                print(f"备份文件中不存在区域: {area_no}")
                continue

            # 转换为更新格式
            transformed_rules = transform_rules_for_restore(rules_of_area)
            objects_to_update.append(transformed_rules)

    # 执行更新
    print(f"\n需要更新的区域数量: {len(objects_to_update)}")
    update_delivery_fee_rules(args.env, token_str, objects_to_update, dry_run=args.dry_run)

    # 输出无需修改的区域
    if no_need_to_modify:
        print(f"\n无须改动的区域数量: {len(no_need_to_modify)}")
        for obj in no_need_to_modify:
            print(f"  无须改动: {obj.get('businessId')}")


if __name__ == "__main__":
    main()
