import requests
from bs4 import BeautifulSoup
import bs4
import json

url = "https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/II?page=1&api_key=RGAPI-daeab076-5b55-4055-b48d-c8a42ddd1498"
url1 = "https://s3-us-west-1.amazonaws.com/riot-developer-portal/seed-data/matches1.json"


ss = {"matches":[{"gameId":2585565774,"platformId":"NA1","gameCreation":1504030815348}]}
print((ss["matches"][0]["gameId"]))
print(json.dumps(ss, sort_keys=True, indent=2)) # 排序并且缩进两个字符输出

# try:
#     r = requests.get(url1)
#     #r.raise_for_status()
#     #r.encoding = r.apparent_encoding
#     text1 = json.loads(r.text)
#     print(type(text1["matches"].gameId))
# except:
#     print(" ")