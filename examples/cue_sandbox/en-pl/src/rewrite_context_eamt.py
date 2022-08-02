import re
from tqdm import tqdm

#for split in ["train", "dev", "test"]:
for split in ["train"]: 
  with open(f'examples/cue_sandbox/en-pl/data/en-pl.{split}.bpe.cxt') as f:
    lines = f.read().splitlines()
  
  new_data = {
    "formality": [],
    "speaker": [],
    "interlocutor": []
  }
  
  for line in tqdm(lines):
    if "singular" in line or "plural" in line:
      template = "Talking to a group of @@people" if "plural" in line else "Talking to a @@person"
      for gender in ["feminine", "masculine", "mixed"]:
        if f"il:{gender}" in line:
          template = re.sub("@@", f"{gender} gender ", template)
      if "@@" in template: # not replaced by any of the rules above
        template = re.sub("a @@", "one ", template)
        template = re.sub("@@", "", template)
      new_data["interlocutor"].append(template)
    else:
      new_data["interlocutor"].append("") 
  
    if "sp:" in line:
      template = "I am a speaker of @@ gender"
      for gender in ["feminine", "masculine"]:
        if f"sp:{gender}" in line:
          new_data["speaker"].append(re.sub("@@", gender, template))
    else:
      new_data["speaker"].append("")
  
    if "<formal>" in line:
      new_data["formality"].append("Formal")
    elif "<informal>" in line:
      new_data["formality"].append("Informal")
    else:
      new_data["formality"].append("")
  
  for cxt in new_data.keys():
    with open(f'examples/cue_sandbox/en-pl/data/context/{split}.{cxt}.cxt', 'w+') as f:
      for line in new_data[cxt]:
        f.write(line + "\n")
