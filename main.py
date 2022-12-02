# THIS CODE SHOULD BE RUN ON A COLAB NOTEBOOK

# Install and import packages
! pip install datasets
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
dataset = load_dataset("climate_fever")

len(dataset)
dataset["test"][0]
len(dataset["test"])
print(dataset)
print(dataset["test"]["claim"])

for d in dataset["test"]["evidences"][0]:
  print(d["evidence_label"])
  print(d["votes"])
  
# Create variables and copy contents for claims, claim_label, and claim_id
claims=dataset["test"]["claim"]
label=dataset["test"]["claim_label"]
claim_id=dataset["test"]["claim_id"]

# Create dataframe with claims and claim_labels
d={"claims":claims,"labels":label}
df=pd.DataFrame(data=d)

claim_id_list, claim_list, claim_label_list, evidence_label_list, votes_list = [], [], [], [], []


for i in range(len(df)):
  claim_id=dataset["test"]["claim_id"][i]
  claim=dataset["test"]["claim"][i]
  claim_label=dataset["test"]["claim_label"][i]
  for d in dataset["test"]["evidences"][i]:
    claim_id_list.append(claim_id)
    claim_list.append(claim)
    claim_label_list.append(claim_label)
    evidence_label_list.append(d["evidence_label"])
    votes_list.append(d["votes"])
