import json
import sys


with open(sys.argv[1], 'r', encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open(sys.argv[2], 'w', encoding="utf-8") as news_collected:
	json.dump(data,news_collected)