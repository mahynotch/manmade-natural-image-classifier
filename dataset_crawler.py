import requests as re
from bs4 import BeautifulSoup
import os
import lxml

KEYLIST = ["Town"]


def getpage(url) -> BeautifulSoup:
    base_site = url
    page = re.get(base_site, headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'})
    assert page.status_code == 200
    base = BeautifulSoup(page.text, "lxml")
    return base

def getpic(url) -> bytes:
    base_site = url
    page = re.get(base_site, headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'})
    # page.content
    return page.content

def get_pic(key: str, num: int) -> BeautifulSoup:
    return getpage(f"https://cn.bing.com/images/search?q={key}&first={num}")

def get_img_set(KEY: str, num: int):
    soup = get_pic(KEY, num)
    all_img = soup.find_all("img", class_="mimg")
    return list(map(lambda a : getpic(a["src"] if "src" in a.attrs else a["data-src"]) , all_img[0:]))

if __name__ == "__main__":
    for KEY in KEYLIST:
        img_set = []
        for i in range(10):
            img_set += get_img_set(KEY, 1 + i * 35)
        for i in range(len(img_set)):
            os.makedirs(f"./crawler_output/manmade/", exist_ok=True)
            with open(f"./crawler_output/manmade/crawlerout_{KEY}_{i}.jpg", "wb") as file:
                file.write(img_set[i])

