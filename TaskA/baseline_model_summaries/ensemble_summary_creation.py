import os
import json
import pandas as pd
json_list = os.listdir()

master_df = None
for json_file in json_list:
    print(json_file)
    if not (json_file.endswith("json") and json_file.startswith("faithful_summary")):
        continue
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
    print(f"{json_file} done")
master_df.to_csv("faithful_summaries.csv",index=True)
