import json
from scripts.cal_num import VOX_LANGS

data={}
list_name = 'lang_union'

with open("iso_639-1.json", "r") as f:
    iso_6391 = json.load(f)

# with open(f"{list_name}.json", "r") as f:
#     data[list_name] = json.load(f)
data[list_name] = VOX_LANGS

family_count = {}
for l in data[list_name]:
    if l in iso_6391:
        f = iso_6391[l]['family']
    else:
        print(l, data[list_name][l])
        continue

    if f in family_count:
        family_count[f] += 1
    else:
        family_count[f] = 1

# for f in family_count.items():
#     print(f"{f[0]}: {f[1]}")

families = sorted([f for f in family_count])
for f in families:
    print(f"{f}: {family_count[f]}")