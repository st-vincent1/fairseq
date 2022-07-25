import torch
import transformers as ppb  # pytorch-transformers by huggingface
import pickle
import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser

BSZ = 256


class ContextEmbedding():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.tokenizer = ppb.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def produce_embeddings(self, context_dir, prefix='train'):
        """Produce context embeddings given a directory.
        The directory should contain all context files for the given dataset.
        The context files will all be opened and context extracted

        context files should follow the naming convention of {train,dev,test}.{arbitrary_name}.cxt"""

        _dir = glob.glob(f"{context_dir}/{prefix}*")
        print(_dir)
        # Initialise empty tensor to be later replaced with a proper buffer
        all_embeddings = torch.empty([0,0,768]).to(self.device)
        for filepath in _dir:
            with open(filepath) as f:
                sentences = f.read().splitlines()

            # resize buffer
            if all_embeddings.nelement() == 0:
                all_embeddings = torch.empty([len(sentences), 0, 768]).to(self.device)

            # buffer for embeddings
            cls_embeddings = torch.empty([0, 768]).to(self.device)

            for i in tqdm(range(0, len(sentences), BSZ)):
                encoded_input = self.tokenizer(sentences[i:i + BSZ],
                                               add_special_tokens=True,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt').to(self.device)
                # indices of empty sentences
                indices = torch.tensor([i for i, x in enumerate(sentences[i:i+BSZ]) if not x]).to(self.device)
                with torch.no_grad():
                    # Get last hidden states of model and then extract the cls embedding ([0][:,0,:])
                    cls = self.model(encoded_input['input_ids'],
                                     attention_mask=encoded_input['attention_mask'])[0][:, 0, :].to(self.device)

                    # nullify matrices where no context given
                    try:
                        cls = cls.index_fill_(0, indices, 0)
                    except IndexError: # no empty strings found
                        pass

                cls_embeddings = torch.cat((cls_embeddings, cls))
            all_embeddings = torch.cat((all_embeddings, cls_embeddings.unsqueeze(1)), dim=1)
        return all_embeddings


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", default="examples/cue_sandbox/data/context")
    args = parser.parse_args()
    x = ContextEmbedding()
    for prefix in ['dev', 'tst-COMMON', 'train', 'test']:
        try:
            cls_embeddings = x.produce_embeddings(args.path, prefix=prefix)
            with open(f"{os.path.dirname(args.path)}/{prefix}.pkl", 'wb+') as out:
                pickle.dump({'cxt': cls_embeddings}, out)

        except FileNotFoundError:
            print(f"Not found {args.path}. Skipping")
            pass