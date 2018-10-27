# -*- coding: utf-8 -*-

import time
import datetime

# 计算给定时间到现在过了多少月份

def date_to_now_month(text):
    text += "1日"
    end = datetime.datetime.strptime(text, '%Y年%m月%d日')
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S')
    delta = start - end
    return (int)(delta.days/30)