from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import time
import sys
import re
import os
import csv
import numpy as np
# print(sys.path)
# help(webdriver)



def getLandL(add):

    for no in range(3):
        options = Options()
        options.add_argument('--headless')    # 不打开浏览器
        options.add_argument('--disable-gpu')  # 禁用GPU硬件加速
        options.add_argument(
            'user-agent="Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1"')  # 添加访问头
        # options.add_argument('proxy-server="60.13.42.109:9999"')    # 添加代理
        driver = webdriver.Chrome(options=options)  # 使用驱动配置
        driver.set_window_size(1920, 1080)
        driver.get("https://www.google.com.hk/maps/place/" + add)
        driver.implicitly_wait(15)  # 等待时间

        L1Mark, L2Mark = .0, .0
        for timerOut in range(10):
            time.sleep(1)
            getUrl = driver.current_url
            element = re.findall("https://www.google.com.hk/maps/place/.*?/.*?(\d+.\d+).*?(\d+.\d+)", getUrl, re.S)
            # print(getUrl)
            # print(element)



            if element != []:
                if no >= 1 and timerOut >= 7:
                    L1Mark, L2Mark = 1., 1.

                if (L1Mark == .0 and L2Mark == .0):
                    L1Mark, L2Mark = float(element[0][0]), float(element[0][1])
                    driver.find_element_by_xpath("//*[@id=\"searchbox-searchbutton\"]").click()
                else:
                    L1, L2 = float(element[0][0]), float(element[0][1])
                    if (L1Mark != L1 and L2Mark != L2):
                        driver.close()
                        return L1, L2

        driver.close()
        time.sleep(20)

    return ["", ""]


class GMCrawler():
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path

    def __getFiles(self,path, suffix = ""):
        return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if
                file.endswith(suffix)]

        # 寫入

    def __myWriteLine(self, rootPath, filename, line, model="a+"):
        # print("filename",filename)

        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        with open(os.path.join(rootPath, filename + ".csv"), model, newline="", encoding="utf-8") as f:
            csv_write = csv.writer(f)
            x = np.array(line)
            if x.ndim == 1:
                csv_write.writerow(line)
            if x.ndim == 2:
                for l in line:
                    csv_write.writerow(l)


    def run(self):

        print(self.input_path)
        for file in self.__getFiles(self.input_path,".csv"):
            print(file)
            with open(file, "r", newline="", encoding="utf-8") as f:
                csv_reader = csv.reader(f)

                try:
                    titleList = next(csv_reader)
                    self.__myWriteLine(self.output_path, os.path.basename(file)[:-4], titleList, model="w")

                    for line in csv_reader:
                        print(line[1] + " " + line[2]+ " " + line[14])
                        line.extend(getLandL(line[1] + " " + line[2]+ " " + line[14]) )

                        self.__myWriteLine(self.output_path, os.path.basename(file)[:-4], line)

                except Exception as e :
                    print("error",e)



#
# if __name__ == '__main__':
#     GMC = GMCrawler("E:\[2021.01.18]HK data check\香港數據爬蟲\[11,May 2021]CrawlerHouingData","E:\[2021.01.18]HK data check\香港數據爬蟲")
#     GMC.run()
