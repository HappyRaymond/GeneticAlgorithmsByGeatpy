import os
import re
from bs4 import BeautifulSoup
# from pyquery import PyQuery as pq
import requests
import json
import pandas as pd
import csv
import time

from multiprocessing import Pool, Manager
import time, random
import numpy as np

class HKDATACrawler():
    def __init__(self,output_path):
        # 11,May 2021 Upgrade  更新數據就更改這個

        self.FILENAME = ("香港", "九龍", "新界-離島")
        self.PAGES = (47610, 59198, 114339)

        # test
        self.rootPath = output_path


        self.titleList = ["date", "region", "Housing estates", "Clinch a deal valence",
                     "Get started to make the corrosion", "Construction area", "actual area",
                     "Construction average price",
                     "actual average price"]

        self.url = "https://www.28hse.com/transaction_data/doaction"
        self.ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
        self.cookie = "__cfduid=d1ec94b6c7afca20ae893ef4abf0b030b1612622553; OAID=89323ea2855454cf31072887c506491b; _ga=GA1.2.1361258515.1612622649; PHPSESSID=gi20li2ovh3uidvls8f67clgj9; OACBLOCK=9.1612660995_7.1612661024_11.1613379823; OACCAP=9.5_7.5_11.1; _gid=GA1.2.1155273326.1614955638; _gat_gtag_UA_1075792_1=1"
        self.origin = "https://www.28hse.com/transaction_data/doaction"
        self.content_type = "multipart/form-data; boundary=----WebKitFormBoundaryYXxGtGiAw6x59Za0"
        self.headers = {"cookie": self.cookie,
                   "user-agent": self.ua,
                   #          "origin":origin ,
                   #         "accept":"application/json, text/javascript, */*; q=0.01",
                   #          "accept-language":"en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                   #          "accept-encoding": "gzip, deflate, br",
                   #         "content-type":content_type
                   }



        self.__postHousingDict = {}
    def postHousing(self,url):
        # 房屋详细信息
        #     url = "https://www.28hse.com/transaction_data/hk/shaukeiwan/22711_monti"

        response = requests.get(url,
                                headers=self.headers,
                                timeout=60)
        soup = BeautifulSoup(response.text, "lxml")
        soupInfo = soup.find_all(name='div', attrs={'class': 'five wide column'})

        retList = []

        if response.status_code == 200 and len(soupInfo) >= 10:

            for i in range(10):
                retList.append(soupInfo[i].string.strip())
        else:
            for i in range(10):
                print("this url have notinfomation")
                retList.append("")

        return retList


    # 查找詳細信息
    def postItems_total(self,cat_id, arearaw, deal_id):
        from_data = {'action': 'get_transaction_data_history',
                     'cat_id': cat_id,
                     'arearaw': arearaw,
                     'deal_id': deal_id}

        total_r = requests.post("https://www.28hse.com/transaction_data/doaction_detail",
                                headers=self.headers,
                                data=from_data,
                                timeout=60)

        responseJson = json.loads(total_r.text)
        if (total_r.status_code == 200):
            return responseJson["data"]["results"]["transaction_history_detail"]["items_total"]
        else:
            return -1

    # 寫入
    def myWriteLine(self,rootPath,filename, line, model="a+"):
        # print("filename",filename)

        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        with open(os.path.join(rootPath,filename + ".csv") , model, newline="", encoding="utf-8") as f:
            csv_write = csv.writer(f)
            x = np.array(line)
            if x.ndim == 1:
                csv_write.writerow(line)
            if x.ndim == 2:
                for l in line :
                    csv_write.writerow(l)

    def get28hse(self,area_id, page):
        # print("area_id {}, page {}".format(area_id, page))
        post_form = {
            'action': 'get_transaction_data_area',
            'area_id': str(area_id),
            'page': str(page)
        }

        response = requests.post(self.url,
                                 headers=self.headers,
                                 data=post_form,
                                 #                                  verify=False,
                                 timeout=60)

        responseJson = json.loads(response.text)
        contextStr = responseJson['data']['results']['html']
        # print(contextStr)

        contextList = re.findall("<tbody>(.*?)</tbody>", contextStr, re.S)
        contextList = re.findall("<tr>(.*?)</tr>", contextList[0], re.S)
        #         print(len(contextList))

        writeList = []
        for line in contextList:
            soup = BeautifulSoup(line, "lxml")
            #     print(doc_1.find('td'))
            rangeRet = line.find("ui red arrow down small icon" )  # "ui red arrow down small icon"  下降 icon 使用負數代替  "ui green arrow up small icon" 上升 正數

            infoLine = []
            tags = [tag for tag in soup.find_all('td')]
            infoLine.append(tags[0].string.strip())
            infoLine.append(tags[1].find('a').string.strip())
            infoLine.append(tags[2].find('a').string.strip())
            infoLine.append(tags[3].string.strip())

            rangeNum = tags[4].get_text().strip()
            if rangeRet != -1:
                rangeNum = "-" + rangeNum
            infoLine.append(rangeNum)

            infoLine.append(tags[5].get_text().replace('\n', '').replace(' ', '').replace('建', '').split("實")[0])
            infoLine.append(tags[5].get_text().replace('\n', '').replace(' ', '').replace('建', '').split("實")[1])
            infoLine.append(tags[6].get_text().replace('\n', '').replace(' ', '').replace('建', '').split("實")[0])
            infoLine.append(tags[6].get_text().replace('\n', '').replace(' ', '').replace('建', '').split("實")[1])
            infoLine.append(tags[7].string.strip())
            infoLine.append(tags[8].string.strip())
            infoLine.append(self.postItems_total(tags[9].find('i')['cat_id'],
                                            tags[9].find('i')['arearaw'],
                                            tags[9].find('i')['deal_id']))

            if tags[2].find('a')['href'] not in self.__postHousingDict.keys():
                self.__postHousingDict[tags[2].find('a')['href']] = self.postHousing(tags[2].find('a')['href'])

            infoLine = infoLine + self.__postHousingDict[tags[2].find('a')['href']]
            #         myWriteLine(FILENAME[fileIndex], infoLine)

            # print("infoLine",infoLine)
            writeList.append(infoLine)
        response.close()
        del (response)

        return writeList



    def runFunction(self,fileIndex, page , lock = None):
        rootPath = self.rootPath

        try:
            writeLine = self.get28hse(fileIndex, page)
            with lock:
                self.myWriteLine(rootPath, self.FILENAME[fileIndex-1], writeLine)

            p = round((page) * 100 / self.PAGES[fileIndex])
            duration = round((time.time() // 100) - self.st, 2)
            remaining = round(duration * 100 / (0.01 + p) - duration, 2)
            print(self.FILENAME[fileIndex-1],
                  " 进度:{0}%，已耗时:  {1}day   {2}min   {3}sec，预计剩余时间:{4}day   {5}min   {6}sec".format(p,
                                                                                                   duration // 3600,
                                                                                                   duration // 60 % 60,
                                                                                                   round(
                                                                                                       duration % 60,
                                                                                                       2),
                                                                                                   remaining // 3600,
                                                                                                   remaining // 60 % 60,
                                                                                                   round(
                                                                                                       remaining % 60,
                                                                                                       2)),
                  end="\r")

        except Exception as e:
            for conut in range(10):
                try:
                    writeLine = self.get28hse(fileIndex, page)
                    with lock:
                        self.myWriteLine(rootPath, self.FILENAME[fileIndex-1], writeLine)
                    break
                except:
                    # print("error page:", page)
                    continue

            print("error page:", page)
            # 这个是输出错误类别的，如果捕捉的是通用错误，其实这个看不出来什么
            print('str(Exception):\t', str(Exception))  # 输出  str(Exception):	<type 'exceptions.Exception'>
            # 这个是输出错误的具体原因，这步可以不用加str，输出
            print('str(e):\t\t', str(e))  # 输出 str(e):		integer division or modulo by zero
            print('repr(e):\t', repr(e))  # 输出 repr(e):	ZeroDivisionError('integer division or modulo by zero',)
            print('traceback.print_exc():')
            # 以下两步都是输出错误的具体位置的
            # traceback.print_exc()
            # print('traceback.format_exc():\n%s' % traceback.format_exc())
            self.myWriteLine(rootPath, "error_" + self.FILENAME[fileIndex-1], ["error page:", page,
                                                                               'str(Exception):', str(Exception),
                                                                               'repr(e):', repr(e)]
                                                                                )



    # 多線程爬取數據
    def run(self):
        pp = Pool(10)  # 进程池的最大数量
        lock = Manager().Lock()

        # 遍歷
        for fileIndex in range(len(self.FILENAME)):
            fileIndex = fileIndex + 1
            self.myWriteLine(self.rootPath, self.FILENAME[fileIndex - 1], self.titleList, model="w")  # 寫入數據文件的頭
            self.st = (time.time() // 100)


            for page in range(self.PAGES[fileIndex - 1]):
                # print("{} {}".format(self.FILENAME[fileIndex - 1],self.PAGES[fileIndex - 1]))
                page = page + 1
                pp.apply_async(self.runFunction, args=(fileIndex,page,lock))
                # self.runFunction(fileIndex,page,lock)
        print("任務添加完畢")
        pp.close()
        # 父进程将被阻塞，等子进程全部执行完成之后，父进程再继续执行
        pp.join()
        print('父进程结束')





#
#
# if __name__ == '__main__':
#     crawler = HKDATACrawler()
#     crawler.run()
#
#
