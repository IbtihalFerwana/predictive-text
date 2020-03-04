import requests
import re
import time
from bs4 import BeautifulSoup
import pandas as pd

def get_urls():
    allblogs = requests.get("https://blogs.aljazeera.net/allblogs")
    soup = BeautifulSoup(allblogs.text,"html.parser")
    article = soup.find_all("article")
    pattern = r'<a\shref="(.*)">'
    p = re.compile(pattern)
    articles = p.findall(str(article))
    articles_set = set(articles)
    urls_set = ['https://blogs.aljazeera.net/'+x for x in list(articles_set)]
    return urls_set

def extract_content(urls_set):
    all_data = []
    for url in urls_set:
        req = requests.get(url)
        print(req)
        req_soup = BeautifulSoup(req.text,"html.parser")

        # extract keywords
        target = req_soup.find("div", {"class":"keyWord_Contaner"})
        pattern_target = r'<a\shref=".*>(.*)</a>'
        p_t = re.compile(pattern_target)
        keywords = ",".join(p_t.findall(str(target)))

        # extract paragraphs
        s = req_soup.find_all("p",{"dir":"RTL"})
        text = " ".join([ss.text.replace("\n","").strip() for ss in s])
        data = {"url":url,
                   "article":text, 
                   "keywords":keywords}

        all_data.append(data)
        time.sleep(2)
    return all_data
def save_to_file(all_data, file_name_csv):
    df = pd.DataFrame(all_data)
    df.to_csv(file_name_csv)