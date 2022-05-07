import json
import requests

def get_keys():
    url = "http://ddragon.leagueoflegends.com/cdn/9.9.1/data/en_US/champion.json"

    r = requests.get(url)
    r.raise_for_status()
    mydata = r.json()
    mylists = mydata['data'].items()

    mydict = {}
    stu = []
    for key,values in mylists:
        stu.append(values['key'])

    for i in range(0,143):
        mydict.update({stu[i]:i})
    
    return(mydict)
