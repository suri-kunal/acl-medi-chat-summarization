import os
import json
import pandas as pd
json_list = os.listdir()

master_df = None
for json_file in json_list:
	if not json_file.endswith("json"):
		continue
	print(json_file)
	with open(json_file,"r") as f:
		a = json.load(f)
	a = a["all"]
	for k,v in a.items():
		a[k] = [v]
	a1 = pd.DataFrame.from_dict(a,orient="columns")
	a1 = a1.rename(index={0:json_file.split("/")[-1]})
	if master_df is None:
		master_df = a1
	else:
		master_df = pd.concat([master_df,a1],axis=0)
master_df.to_csv("model_summaries.csv",index=True)
