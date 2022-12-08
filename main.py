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

a = 0
b = 0
c = 0
d = 0

a1 = 0
b1 = 0
c1 = 0
d1 = 0

labels = []
length = []

supports = []
refutes = []
not_enough_info = []
disputed = []

for i in range(len(df)):
  if(label[i] == 0):
    # Supports
    a = a + len(dataset["test"]["claim"][i])
    a1 = a1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(0)
    supports.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 1):
    # Refutes
    b = b + len(dataset["test"]["claim"][i])
    b1 = b1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(1)
    refutes.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 2):
    # Not enough information
    c = c + len(dataset["test"]["claim"][i])
    c1 = c1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(2)
    not_enough_info.append(len(dataset["test"]["claim"][i]))
  if(label[i] == 3):
    # Disputed
    d = d + len(dataset["test"]["claim"][i])
    d1 = d1 + 1
    length.append(len(dataset["test"]["claim"][i]))
    labels.append(3)
    disputed.append(len(dataset["test"]["claim"][i]))

print(a, ": ", a1)
print(b, ": ", b1)
print(c, ": ", c1)
print(d, ": ", d1)

print("Supports: ", 83754/654)
print("Refutes: ", 27698/253)
print("Not Enough Information: ", 59060/474)
print("Disputed: ", 18909/154)

plt.hist(supports, alpha = 0.5)
plt.hist(not_enough_info, alpha = 0.5)
plt.hist(disputed, alpha = 0.5)
plt.hist(refutes, alpha = 0.5)

plt.title('Frequency of Claim Length')
plt.xlabel('Claim Length (# of characters)')
plt.ylabel('Number of Occurences')

legend = ['Supports', 'Refutes', 'NEI', 'Disputes']
plt.legend(legend)

plt.style.use('seaborn')
plt.title('Claim Length vs. Claim Label')
plt.xlabel('Claim Length (# of characters)')
plt.ylabel('Claim Label (0, 1, 2, 3)')
plt.scatter(length, labels, edgecolor = 'black', linewidth = 1, alpha = 0.2)

# Create a data frame with claim_id, claim, claim_label, evidence_label, and votes
data={'claim_id' : claim_id_list, 'claim' : claim_list, 'claim_label' : claim_label_list, 'evidence_label' : evidence_label_list, 'votes' : votes_list}
df_evidence = pd.DataFrame(data=data)

# Print first five rows of the data frame
df_evidence.head()

print(votes_list)

df.labels.value_counts()

# Create a table with claim_id and claim_label
data_labels = {
    'Claim ID' : claim_id,
    'Claim Label' : label
}
data_table = pd.DataFrame(data_labels)
data_table

# Create new variable with length of claim text for each claim and plot
claim_lens=[len(text) for text in df.claims.values]
plt.hist(claim_lens)
