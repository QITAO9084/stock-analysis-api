#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球历史数据爬虫 - 多数据源
数据源1: 中国福利彩票官网 API
数据源2: 新浪彩票API
数据源3: 腾讯彩票API
"""
import requests
import json
import time
import re
from datetime import datetime

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.cwl.gov.cn/',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

def try_sina_api():
    """新浪彩票API - 双色球历史数据"""
    url = "https://kaijiang.sina.com.cn/lottery/ajax_get_kj.php?lottery_type=ssq&page=1&page_size=200"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"新浪API: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            d = r.json()
            print("新浪返回:", json.dumps(d, ensure_ascii=False)[:300])
            return d
    except Exception as e:
        print(f"新浪API异常: {e}")
    return None

def try_cwl_org_cn():
    """中国福利彩票官网 - 双色球开奖公告"""
    # 福彩官网API
    url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    params = {
        'name': 'ssq',
        'issueCount': 200,
        'issueStart': '',
        'issueEnd': '',
        'dayStart': '',
        'dayEnd': '',
    }
    try:
        r = requests.get(url, params=params, headers={**headers, 'Referer': 'https://www.cwl.gov.cn/'}, timeout=10)
        print(f"福彩官网API: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            d = r.json()
            print("福彩返回:", json.dumps(d, ensure_ascii=False)[:300])
            return d
    except Exception as e:
        print(f"福彩官网API异常: {e}")
    return None

def try_tencent_api():
    """腾讯彩票API"""
    url = "https://cp.sogou.com/lottery/ssq.html"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"腾讯彩票: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            # 找JSON数据
            match = re.search(r'var\s+data\s*=\s*({.*?});', r.text, re.DOTALL)
            if match:
                print("腾讯彩票找到data变量")
                return json.loads(match.group(1))
    except Exception as e:
        print(f"腾讯彩票异常: {e}")
    return None

def try_500com_ajax():
    """500彩票网AJAX接口"""
    # 500.com的历史数据通过AJAX加载
    url = "https://datachart.500.com/ssq/newinc/history.php"
    params = {
        'start': '24001',
        'end': '24200',
    }
    try:
        r = requests.get(url, params=params, headers={**headers, 'Referer': 'https://datachart.500.com/ssq/'}, timeout=15)
        print(f"500彩票: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            # 解析HTML表格
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', id='tablelist')
            if table:
                print("500彩票找到tablelist表格")
                return table
    except Exception as e:
        print(f"500彩票异常: {e}")
    return None

def try_lehecai():
    """乐和彩API"""
    url = "https://www.lehecai.com/lottery/kaijiang/ssq.html"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"乐和彩: {r.status_code}, len={len(r.text)}")
        if r.status_code == 200:
            print("乐和彩页面前200字:", r.text[:200])
    except Exception as e:
        print(f"乐和彩异常: {e}")
    return None

def try_api68_ssq():
    """api68.com 双色球专用接口"""
    # lotCode=10037是排列3，双色球是10001?
    for code in ['10001', '10002', '10003']:
        url = f"https://api.api68.com/pks/getPksHistoryList.do?lotCode={code}"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            print(f"api68 lotCode={code}: {r.status_code}")
            if r.status_code == 200:
                d = r.json()
                print(f"  返回: {json.dumps(d, ensure_ascii=False)[:200]}")
        except Exception as e:
            print(f"api68异常: {e}")

if __name__ == '__main__':
    print("=== 测试各数据源 ===")
    print("\n1. 新浪彩票API:")
    try_sina_api()
    
    print("\n2. 福彩官网API:")
    try_cwl_org_cn()
    
    print("\n3. api68.com 双色球:")
    try_api68_ssq()
    
    print("\n4. 乐和彩:")
    try_lehecai()
    
    print("\n=== 测试完成 ===")
