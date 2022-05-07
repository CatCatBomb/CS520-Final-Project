import requests
import json

url = "http://ddragon.leagueoflegends.com/cdn/9.9.1/data/en_US/champion.json"
try:
    r = requests.get(url)
    r.raise_for_status()
    mydata = r.json()
    print(json.dumps(mydata['data'].items(), sort_keys=True, indent=2))
except:
    print("error")
