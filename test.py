import json
import torch
import copy
from dataset import readJSON, writeJSON
import toolForward

FP = "C:/CHI_design_tool/conduit.json"
batch_FP = "C:/CHI_design_tool/batch/conduit.json"
single = readJSON(FP)
use = FP

# batch = []
# for i in range(100):
#     batch += [copy.deepcopy(single)]
# writeJSON(batch_FP, batch)

batch_files = readJSON(use)
print(len(batch_files))

toolForward.main(use)

batch_files = readJSON(use.replace(".json", "_result.json"))
print(len(batch_files))