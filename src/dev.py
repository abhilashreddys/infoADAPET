import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.adapet import adapet
from src.utils.Config import Config
from src.eval.eval_model import dev_eval
from src.utils.util import device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-d', "--dataset",default="")
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)
    config.eval_dev = True
    config.eval_train = False

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()
    
    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "epoch_"+str(config.epochs-1)+"_"+config.pretrained_weight+"_best_model.pt")))
    dev_acc, dev_logits = dev_eval(config, model, batcher, 0, spl = str(args.dataset))

    print("Dev Acc: %.3f" % (dev_acc) + '\n')
    np.savetxt(os.path.join(config.exp_dir, 'dev_'+str(args.dataset)+'.txt'),np.argmax(dev_logits,axis=1))
    # with open(os.path.join(config.exp_dir, "dev_logits.npy"), 'wb') as f:
    #     np.save(f, dev_logits)
