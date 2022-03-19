import argparse
import os
import torch
import numpy as np
from transformers import *

from src.data.Batcher import Batcher
from src.utils.Config import Config
from src.utils.util import device
from src.adapet import adapet
from src.eval.eval_model import test_eval,test_alpha_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-a', "--alpha", required=True)
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()

    model = adapet(config, tokenizer, dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "epoch_"+str(config.epochs-1)+"_"+config.pretrained_weight+"_best_model.pt")))
    # test_eval(config, model, batcher)
    test_alpha_acc, test_alpha_logits = test_alpha_eval(config, model, batcher,alpha=int(args.alpha))
    print("Test alpha_%s Acc: %.3f" % (args.alpha,test_alpha_acc) + '\n')
    np.savetxt(os.path.join(args.exp_dir, 'test_alpha'+str(args.alpha)+'_logits.txt'),test_alpha_logits)
    np.savetxt(os.path.join(args.exp_dir, 'test_alpha'+str(args.alpha)+'.txt'),np.argmax(test_alpha_logits,axis=1))

