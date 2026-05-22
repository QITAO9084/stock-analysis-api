#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球历史数据爬虫 - 基于福彩官网API
爬取福彩官网公开数据，存储到 ssq_history.json
"""
import requests
import json
import time
from datetime import datetime

API_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.cwl.gov.cn/',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest',
}

def fetch_page(page_no, page_size=50):
    """获取单页数据"""
    params = {
        'name': 'ssq',
        'issueCount': '',
        'issueStart': '',
        'issueEnd': '',
        'dayStart': '',
        'dayEnd': '',
        '_': int(time.time() * 1000),
    }
    try:
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  请求失败: {e}")
    return None

def parse_result(result_list):
    """解析福彩官网返回的数据"""
    records = []
    for item in result_list:
        try:
            period = item.get('code', '').strip()
            date = item.get('date', '').strip()
            red_str = item.get('red', '').strip()
            blue_str = item.get('blue', '').strip()
            if not period or not red_str:
                continue
            red = [int(x) for x in red_str.split(',') if x.strip()]
            blue = int(blue_str) if blue_str else 0
            if len(red) == 6 and blue:
                records.append({
                    'period': period,
                    'red': sorted(red),
                    'blue': blue,
                    'date': date,
                })
        except Exception as e:
            print(f"  解析失败: {e}, item={item}")
            continue
    return records

def crawl_history(target_count=500):
    """爬取历史数据"""
    print(f"开始爬取双色球历史数据，目标: {target_count}期")
    print(f"API: {API_URL}")
    
    all_records = []
    page_no = 1
    page_size = 30  # 福彩API每页30条
    
    while len(all_records) < target_count:
        print(f"\n正在获取第{page_no}页...", end=' ')
        data = fetch_page(page_no, page_size)
        
        if not data or data.get('state') != 0:
            print(f"失败, state={data.get('state') if data else 'None'}")
            break
        
        result_list = data.get('result', [])
        if not result_list:
            print("无数据")
            break
        
        records = parse_result(result_list)
        if not records:
            print("解析失败")
            break
            
        all_records.extend(records)
        print(f"获取{len(records)}条，累计{len(all_records)}条")
        
        # 检查是否已拿到足够数据
        if len(all_records) >= target_count:
            break
            
        page_no += 1
        time.sleep(0.8)  # 避免请求过快
        
    # 去重（按period）
    seen = set()
    unique = []
    for r in all_records:
        if r['period'] not in seen:
            seen.add(r['period'])
            unique.append(r)
    
    # 按期号排序（新→旧）
    unique.sort(key=lambda x: x['period'], reverse=True)
    
    print(f"\n去重后: {len(unique)}期")
    if unique:
        print(f"最早: {unique[-1]['period']} ({unique[-1]['date']})")
        print(f"最新: {unique[0]['period']} ({unique[0]['date']})")
    
    return unique

def merge_with_existing(new_data, existing_file='ssq_history.json'):
    """与现有数据合并"""
    try:
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"\n现有数据: {len(existing)}期")
    except Exception:
        existing = []
    
    existing_periods = {r['period'] for r in existing}
    added = 0
    for r in new_data:
        if r['period'] not in existing_periods:
            existing.append(r)
            existing_periods.add(r['period'])
            added += 1
    
    # 按期号排序（新→旧）
    existing.sort(key=lambda x: x['period'], reverse=True)
    
    print(f"新增: {added}期")
    print(f"合并后: {len(existing)}期")
    return existing

def save_data(data, filename='ssq_history.json'):
    """保存数据"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存到 {filename}")

if __name__ == '__main__':
    # 先测试一页
    print("=== 测试API连通性 ===")
    test = fetch_page(1, 30)
    if test:
        print(f"state: {test.get('state')}")
        print(f"message: {test.get('message')}")
        result = test.get('result', [])
        print(f"第一页数据条数: {len(result)}")
        if result:
            print(f"示例: period={result[0].get('code')}, red={result[0].get('red')}, blue={result[0].get('blue')}, date={result[0].get('date')}")
    
    if not test or test.get('state') != 0:
        print("\nAPI测试失败，请检查网络或API地址")
    else:
        print("\n=== 开始爬取历史数据 ===")
        new_data = crawl_history(target_count=500)
        if new_data:
            merged = merge_with_existing(new_data)
            save_data(merged)
            print("\n=== 完成 ===")
