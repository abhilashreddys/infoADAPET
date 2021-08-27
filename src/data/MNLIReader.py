import os
import json
import random
import csv
import re
import sys
import string
import numpy as np
import itertools
import torch
from collections import defaultdict
# from datasets import load_dataset, ClassLabel
# from src.data.cleaner import preprocessTxt
from src.data.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils.util import device

class MNLIReader(object):
    '''
        https://cims.nyu.edu/~sbowman/multinli/paper.pdf
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.list_true_lbl = []

        self.num_lbl = 3

        self.pet_labels = [["No", "Maybe", "Yes"]]
        self.pet_patterns = [["[HYPOTHESIS] ? [SEP]", " {}, ".format(self.tokenizer.mask_token), "[PREMISE] [SEP]"],
                             ["\" [HYPOTHESIS] \" ? [SEP]", " {}, ".format(self.tokenizer.mask_token), "\" [PREMISE] \" [SEP]"],
                             ["[HYPOTHESIS] ? [SEP]", " {}. ".format(self.tokenizer.mask_token), "[PREMISE] [SEP]"],
                             ["\" [HYPOTHESIS] \" ? [SEP]", " {}. ".format(self.tokenizer.mask_token), "\" [PREMISE] \" [SEP]"],
                             ["[HYPOTHESIS] ? [SEP]", " {}. ".format(self.tokenizer.mask_token), "[PREMISE] [SEP]"],
                             ["\" [HYPOTHESIS] \" ? [SEP]", " {}. ".format(self.tokenizer.mask_token), "\" [PREMISE] \" [SEP]"]]

        self.pet_pvps = list(itertools.product(self.pet_patterns, self.pet_labels))
        self._num_pets = len(self.pet_pvps)
        self._pet_names = ["PET{}".format(i+1) for i in range(self._num_pets)]

        # self.dict_lbl_2_idx = {"Contradiction": 0, "Neutral": 1, "Entailment":2}

        self.dict_inv_freq = defaultdict(int)
        self.tot_doc = 0

    def _get_file(self, split):
        '''
        Get filename of split

        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("src/data", "mnli", "mnli_train_sampled10k.tsv")
        elif split.lower() == "dev":
            file = os.path.join("src/data", "mnli", "mnli_dev_sampled150.tsv")
        else:
            raise NotImplementedError("No file for other splits")
        return file

    def read_dataset(self, split=None, is_eval=False):
        '''
        Read the dataset

        :param split: partition of the
        :param is_eval:
        '''

        file = self._get_file(split)
        data = []

        with open(file,'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if(i == 0): # header
                    continue
                else:
                    dict_input = {}
                    dict_input["premise"] = line[2]
                    dict_input["hypothesis"] = line[5]
                    dict_input["idx"] = str(line[1])
                    dict_output = {}
                    dict_output["lbl"] = int(line[9])
                    dict_input_output = {"input": dict_input, "output": dict_output}
                    data.append(dict_input_output)
        return data
    @property
    def pets(self):
        return self._pet_names

    def get_num_lbl_tok(self):
        return 1

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_hypothesis = batch["input"]["hypothesis"]
        list_premise = batch["input"]["premise"]

        list_input_ids = []
        bs = len(batch["input"]["premise"])
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        for b_idx, (h, p) in enumerate(zip(list_hypothesis, list_premise)):
            mask_txt_split_tuple = []
            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                mask_txt_split_inp = txt_split.replace("[HYPOTHESIS]", h).replace("[PREMISE]", p)
                mask_txt_split_tuple.append(mask_txt_split_inp)

                # Trim the paragraph
                if "[PREMISE]" in txt_split:
                    txt_trim = idx

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx,:self.get_num_lbl_tok()] = range(mask_idx, mask_idx+self.get_num_lbl_tok())

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), label

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):

        '''
        Prepare for train

        :param batch:
        :return:
        '''

        list_hypothesis = batch["input"]["hypothesis"]
        list_premise = batch["input"]["premise"]

        bs = len(batch["input"]["hypothesis"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        list_orig_input_ids = []
        list_masked_input_ids = []

        for b_idx, (h, p, lbl) in enumerate(zip(list_hypothesis, list_premise, prep_lbl)):
            txt_split_tuple = []

            txt_trim = -1

            for idx, txt_split in enumerate(pattern):
                txt_split_inp = txt_split.replace("[HYPOTHESIS]", h).replace("[PREMISE]", p).replace("[MASK]", label[lbl])

                txt_split_tuple.append(txt_split_inp)

                # Trim the paragraph
                if "[PREMISE]" in txt_split:
                    txt_trim = idx

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)

    def prepare_eval_pet_mlm_batch(self, batch, mode="PET1"):
        '''
        Prepare for train

        :param batch:
        :return:
        '''
        list_hypothesis = batch["input"]["hypothesis"]
        list_premise = batch["input"]["premise"]

        list_input_ids = []
        list_masked_input_ids = []

        pattern, label = self.pet_pvps[self._pet_names.index(mode)]

        for b_idx, (h, p) in enumerate(zip(list_hypothesis, list_premise)):
            mask_idx = None

            for l_idx, lbl in enumerate(label):
                txt_split_tuple = []

                for idx, txt_split in enumerate(pattern):
                    txt_split_inp = txt_split.replace("[HYPOTHESIS]", h).replace("[PREMISE]", p).replace("[MASK]", lbl)
                    txt_split_tuple.append(txt_split_inp)

                    # Trim the paragraph
                    if "[PREMISE]" in txt_split:
                        txt_trim = idx

                orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config,
                                                                                  txt_split_tuple[0],
                                                                                  txt_split_tuple[1],
                                                                                  txt_split_tuple[2], txt_trim,
                                                                                  mask_idx=mask_idx)
                list_input_ids.append(orig_input_ids)
                list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_input_ids).to(device), torch.tensor(list_masked_input_ids).to(device)

    def store_test_lbl(self, list_idx, pred_lbl, true_lbl, logits):
        self.list_true_lbl.append(pred_lbl)

    def flush_file(self, write_file):
        self.list_true_lbl = torch.cat(self.list_true_lbl, dim=0).cpu().int().numpy().tolist()

        read_file = self._get_file("test_alpha1")

        with open(read_file,'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for i, line in enumerate(tsvreader):
                if(i == 0): # header
                    continue
                else:
                    answer_dict = {}
                    answer_dict["idx"] = i
                    answer_dict["label"] = self.list_true_lbl[i]
                    write_file.write(json.dumps(answer_dict) + "\n")