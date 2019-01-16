import numpy as np
import re
from pypinyin import pinyin, Style

num=['零','一','二','三','四','五','六','七','八','九']
kin=['十','百','千','万','零']

def sadd(x):
    x.reverse()
    if len(x) >= 2:
        x.insert(1,kin[0])
        if len(x) >= 4:
            x.insert(3,kin[1])
            if len(x) >= 6:
                x.insert(5,kin[2])
                if len(x) >= 8:
                    x.insert(7,kin[3])
                    if len(x) >= 10:
                        x.insert(9,kin[0])
                        if len(x) >= 12:
                            x.insert(11,kin[1])

    x=fw(x)
    x=d1(x)
    x=d2(x)
    x=dl(x)
    return x

def seperate(digits):
    i=list(digits)
    for j in i:
        i[(i.index(j))]=num[int(j)]
    return ''.join(i)

def combine(digits):
    i=list(digits)
    for j in i:
        i[(i.index(j))]=num[int(j)]
    i=sadd(i)
    if len(i) >= 2 and i[0] == '一' and i[1] == '十':
        return ''.join(i)[1:]
    return ''.join(i)

def combine_with_zero(digits):
    s = combine(digits)
    if s == '':
        return '零'
    return s


def d1(x):
    if '零' in x:
        a=x.index('零')
        if a==0:
            del x[0]
            d1(x)
        else:
            if x[a+2] in ['十','百','千','万','零']:
                if x[a+1] != '万':
                    del x[a+1]
                    d1(x)
    return x
def d2(x):
    try:
        a=x.index('零')
        if x[a-1] in ['十','百','千','零']:
            del x[a-1]
            d2(x[a+1])
    except:pass
    return x

def fw(x):
    if len(x) >= 9:
        if x[8] == '零':
            del x[8]
    return x
def dl(x):
    try:
        if x[0]=='零':
            del x[0]
            del1(x)
    except:pass
    x.reverse()
    x=''.join(x)
    return x

# ---------------------
# 作者：PlusZhang
# 来源：CSDN
# 原文：https://blog.csdn.net/PlusChang/article/details/72991191
# 版权声明：本文为博主原创文章，转载请附上博文链接！




def expand(text):
    # %
    text = re.sub('[0-9]+%', (lambda s: '百分之' + s.group()[:-1]) ,text)

    # float
    text = re.sub('[0-9]+\.[0-9]+', (lambda s: combine_with_zero(s.group().split('.')[0]) + '点' + seperate(s.group().split('.')[1])) ,text)
    # year
    text = re.sub('[1-9]0+年', (lambda s: combine_with_zero(s.group()[:-1]) + '年') ,text)
    text = re.sub('[0-9]+年', (lambda s: seperate(s.group()[:-1]) + '年') ,text)

    # other digits
    text = re.sub('[0-9]+', (lambda s: combine_with_zero(s.group())) ,text)
    return text


def get_pinyin(text):
    t = " ".join(map(lambda s: s[0], pinyin(text, style = Style.TONE3)))
    return re.sub('[a-z]+( |\Z)', lambda s: s.group()+'0' if  s.group()[-1] != ' ' else s.group()[:-1] +'0 ' , t)
    

