import torch
import transformers as ppb  # pytorch-transformers by huggingface
import pickle
import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser
import torch.multiprocessing as mp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

BSZ = 128


class ContextEmbedding:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def embeddings_to_float_storage(self, input_dir, prefix):
        def initialise_buffer(filename, num_samples, num_contexts, embed_dim):
            logging.info(f"--- Initialising buffer with size {num_samples} x {num_contexts} x {embed_dim}")
            binary_buffer = torch.FloatTensor(
                torch.FloatStorage.from_file(filename, shared=True, size=num_samples * num_contexts * embed_dim)) \
                .reshape(num_samples, num_contexts, embed_dim)
            return binary_buffer

        _dir = glob.glob(f"{input_dir}/{prefix}*")
        if not _dir:
            return
        out_filename = f"{os.path.dirname(args.path)}/{prefix}.bin"
        logging.info(f"--- Scrapping data from {_dir} and saving to {out_filename}...")

        bin_buff = None

        for file_idx, filepath in enumerate(_dir):
            # Read sentences from one context file
            with open(filepath) as f:
                sentences = f.read().splitlines()

            # initialise float buffer if not done yet
            if bin_buff is None:
                bin_buff = initialise_buffer(out_filename, num_samples=len(sentences), num_contexts=len(_dir),
                                             embed_dim=768)

            for i in tqdm(range(0, len(sentences), BSZ)):
                encoded_input = self.tokenizer(sentences[i:i + BSZ],
                                               add_special_tokens=True,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt').to(self.device)
                # indices of empty sentences
                indices = torch.tensor([i for i, x in enumerate(sentences[i:i + BSZ]) if not x]).to(self.device)

                with torch.no_grad():
                    # Get last hidden states of model and then extract the cls embedding ([0][:,0,:])
                    cls = self.model(encoded_input['input_ids'],
                                     attention_mask=encoded_input['attention_mask'])[0][:, 0, :].to(self.device)

                    # nullify matrices where no context given
                    try:
                        cls = cls.index_fill_(0, indices, 0)
                    except IndexError:  # no empty strings found
                        pass
                bin_buff[i:i + BSZ, file_idx, :] = cls


        # Alternative (bad) approach: save everything at the very end, having concatenated all tensors to all_embeddings
        # logging.info("--- Binarising tensors...")
        # all_embeddings = list(all_embeddings)
        # for idx in tqdm(range(len(all_embeddings))):
        #     bin_buff[idx] = all_embeddings[idx]

#        samples = torch.FloatTensor(
#            torch.FloatStorage.from_file(out_filename, shared=False, size=len(sentences) * 2 * 768)).reshape(
#            len(sentences),
#            2, 768)
#        dataset = torch.utils.data.TensorDataset(samples)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", default="examples/cue_sandbox/data/context")
    args = parser.parse_args()
    x = ContextEmbedding()
    for prefix in ['train', 'test']:
        try:
            x.embeddings_to_float_storage(args.path, prefix=prefix)
        except FileNotFoundError:
            logging.warning(f"Not found {args.path}. Skipping")
            pass
