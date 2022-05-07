import json
import requests
import csv

url = "https://s3-us-west-1.amazonaws.com/riot-developer-portal/seed-data/matches1.json"

r = requests.get(url)
r.raise_for_status()
data = r.json()
#out = open('t_data1.csv','a',newline='')
#csv_write = csv.writer(out,dialect='excel')
for n in range(0,10):
    stu = [0 for _ in range(143)]
    for i in range(0,10):
        ind = int(data["matches"][n]["participants"][i]["championId"])-1
        print(ind)
    #    stu[ind] = 1
    # for m in range(0,2):
    #     for j in range(0,5):
    #         stu.append(data["matches"][n]["teams"][m]["bans"][j]["championId"])
    stu.append(data["matches"][n]["teams"][0]["win"])
    #csv_write.writerow(stu)
#out.close()

#csv格式
#       pick        |         ban        |     win
#   100      200    |    100       200   |     100