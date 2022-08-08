import torch
import psutil
import transformers as ppb  # pytorch-transformers by huggingface
import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser

import logging
import itertools

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

BSZ = 16384


class ContextEmbedding:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        self.model = ppb.DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def embeddings_to_float_storage(self, input_dir, prefix):
        def initialise_buffer(filename, num_samples, num_contexts, embed_dim):
            logging.info(f"--- Initialising buffer with size {num_samples} x {num_contexts} x {embed_dim}")
            binary_buffer = torch.FloatTensor(
                torch.FloatStorage.from_file(filename, shared=True, size=num_samples * num_contexts * embed_dim)) \
                .reshape(num_samples, num_contexts, embed_dim)
            return binary_buffer

        _dir = glob.glob(os.path.join(input_dir, "context", f"{prefix}*"))
        if not _dir:
            raise FileNotFoundError
        out_filename = os.path.join(input_dir, f"{prefix}.bin")

        if os.path.exists(out_filename):
            logging.warning(f"--- Binarised file for {prefix} already exists. skipping...")
            raise FileNotFoundError
        logging.info(f"--- Scrapping data from {_dir} and saving to {out_filename}...")

        bin_buff = None
        for file_idx, filepath in enumerate(_dir):
            # Read sentences from one context file
            with open(filepath) as f:
                sentences = f.read().splitlines()

            # initialise float buffer if not done yet
            if bin_buff is None:
                out_filename = os.path.join(input_dir, f"{prefix}.bin")
                
                logging.info(f"--- Initialising a buffer...")
                bin_buff = initialise_buffer(out_filename, num_samples=len(sentences), num_contexts=len(_dir),
                                             embed_dim=768)
                logging.info(f"--- Buffer initialised....")

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

        return len(sentences), len(_dir)

    @staticmethod
    def combine_storages(dirname, prefix, data_len, num_contexts, embed_dim):
        full_buffer = torch.FloatTensor(
            torch.FloatStorage.from_file(os.path.join(dirname, f"{prefix}.bin"), shared=True,
                                         size=data_len * num_contexts * embed_dim)) \
            .reshape(data_len, num_contexts, embed_dim)
        logging.info("--- Opened new storage.")
        for i in tqdm(range(num_contexts)):
            logging.info("Loading old samples.")
            samples = torch.FloatTensor(
                torch.FloatStorage.from_file(os.path.join(dirname, f"{prefix}_{i}.bin"), shared=False,
                                             size=data_len * 1 * 768)).reshape(
                data_len, 1, 768)
            print(full_buffer.shape, samples.shape)
            full_buffer[:, i, :] = samples.squeeze()

    @staticmethod
    def delete_partial_files(path, prefix):
        for i in itertools.count():
            try:
                os.remove(os.path.join(path, f"{prefix}_{i}.bin"))
            except OSError:
                return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", default="examples/cue_sandbox/data")
    args = parser.parse_args()
    x = ContextEmbedding()
    for prefix in ['test', 'valid', 'dev', 'tst-COMMON']:
#    for prefix in ['train']:
        try:
            # Save contexts to individual files
            data_len, num_contexts = x.embeddings_to_float_storage(args.path, prefix=prefix)
#            x.combine_storages(args.path, prefix=prefix, data_len=data_len, num_contexts=num_contexts, embed_dim=768)
#            x.delete_partial_files(args.path, prefix=prefix)
        except FileNotFoundError:
            logging.warning(f"Not found {args.path}. Skipping")
            pass
