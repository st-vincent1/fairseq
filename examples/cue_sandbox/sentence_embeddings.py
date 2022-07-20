import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch-transformers by huggingface
import warnings
import pickle
from datasets import Dataset
import glob
import os
from tqdm import tqdm

BSZ = 512


class ContextEmbedding():
    def __init__(self):
        self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def produce_embeddings(self, context_dir, prefix='train'):
        """Produce context embeddings given a directory.
        The directory should contain all context files for the given dataset.
        The context files will all be opened and context extracted

        context files should follow the naming convention of {train,dev,test}.{arbitrary_name}.cxt"""

        _dir = glob.glob(f"{context_dir}/{prefix}*")
        print(_dir)
        all_embeddings = torch.empty([0, 0, 768])
        for filepath in _dir:
            with open(filepath) as f:
                sentences = f.read().splitlines()

            # resize buffer
            if all_embeddings.nelement() == 0:
                all_embeddings = torch.empty([0, len(sentences), 768])

            # buffer for embeddings
            cls_embeddings = torch.empty([0, 768])

            for i in tqdm(range(0, len(sentences), BSZ)):
                encoded_input = self.tokenizer(sentences[i:i + BSZ],
                                               add_special_tokens=True,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt')
                # indices of empty sentences
                indices = torch.tensor([i for i, x in enumerate(sentences[i:i+BSZ]) if not x])
                with torch.no_grad():
                    # Get last hidden states of model and then extract the cls embedding ([0][:,0,:])
                    cls = self.model(encoded_input['input_ids'],
                                     attention_mask=encoded_input['attention_mask'])[0][:, 0, :]
                    # nullify matrices where no context given
                    try:
                        cls = cls.index_fill_(0, indices, 0)
                    except IndexError: # no empty strings found
                        pass

                cls_embeddings = torch.cat((cls_embeddings, cls))

            all_embeddings = torch.cat((all_embeddings, cls_embeddings.unsqueeze(0)))
        return all_embeddings


x = ContextEmbedding()
path = 'examples/cue_sandbox/data/context'
for prefix in ['dev', 'train']:
    cls_embeddings = x.produce_embeddings(path, prefix=prefix)

    with open(f"{os.path.dirname(path)}/{prefix}.pkl", 'wb+') as out:
        pickle.dump({'cxt': cls_embeddings}, out)
