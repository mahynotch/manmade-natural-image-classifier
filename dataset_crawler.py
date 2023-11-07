import requests as re
from bs4 import BeautifulSoup
import os
import lxml
import json

KEYLIST = ["Town", "Grosery shop", "Bridge", "Road", "Car", "Train", "Plane", "Boat", "Subway station", "Bar", "Restaurant", "Cafe", "Construction site", "School", "Factory", "Buildings in Shenzhen", "Castle", "Bunk", "Greenhouse", "Lighthouse"]
# KEYLIST = [ "Park", "Beach", "Mountain", "Forest", "Lake", "River", "Waterfall", "Desert", "Cave", "Volcano", "Flower", "Tree", "Grass", "Sky", "Cloud", "Sun", "Moon", "Star", "Planet", "Galaxy", "Universe"]

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

def get_pic_from_key(key: str, num1: int, num2: int) -> BeautifulSoup:
    return getpage(f"https://cn.bing.com/images/search?q={key}&first={num1}&SFX={num2}")

def get_img_list(KEY: str, num1: int, num2):
    soup = get_pic_from_key(KEY, num1, num2)
    all_img = soup.find_all("a", class_="iusc")
    print(f"Searching first {num1}, SFX {num2} of {KEY}, got {len(all_img)} images.")
    return list(map(lambda a : getpic(json.loads(a["m"])["turl"]), all_img[0:]))

if __name__ == "__main__":
    for KEY in KEYLIST:
        img_set = []
        for i in range(20):
            img_set += get_img_list(KEY, 1 + i * 35, 1 + i)
        img_set = set(img_set)
        for i in range(len(img_set)):
            os.makedirs(f"./crawler_output/manmade/", exist_ok=True)
            with open(f"./crawler_output/manmade/crawlerout_{KEY}_{i}.jpg", "wb") as file:
                file.write(img_set.pop())

