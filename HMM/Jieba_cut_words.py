#!/usr/bin/python
# -*- coding:utf-8 -*-

import jieba.posseg as pseg

if __name__ == '__main__':
    f = open('26.novel.txt','r',encoding='utf-8')
    data = f.read()
    f.close()

    temp = pseg.cut(data)
    for item in temp:
        print(item.word,'|',end='')