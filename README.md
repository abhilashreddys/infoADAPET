# INFOTABS + ADAPET #

[ACL 2020](https://acl2020.org/) paper: [INFOTABS: Inference on Tables as Semi-structured Data](https://www.aclweb.org/anthology/2020.acl-main.210.pdf). To explore the dataset online visit [project page](https://infotabs.github.io).

[Infotabs Dataset](https://github.com/infotabs/infotabs)

This repository contains the modified [official code](https://github.com/rrmenon10/ADAPET) for the paper: "[Improving and Simplifying Pattern Exploiting Training](https://arxiv.org/abs/2103.11955)".

The model improves and simplifies [PET](https://arxiv.org/abs/2009.07118) with a decoupled label objective and label-conditioned MLM objective. 

## How run on colab? ##

Open [colab](https://colab.research.google.com/)
Make a new notebook and connect it to hardware accelerator.

```sh
!git clone https://github.com/abhilashreddys/infoADAPET.git
!pip install crcmod jsonpickle wandb transformers tqdm sentencepiece
cd infoADAPET/
```

Processed INFOTABS data is already in the git repo. \
Download data (if required)
Superglue

```sh
mkdir -p data
cd data
mkdir -p superglue
cd superglue
wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip"
unzip combined.zip
cd ../..
```

Fewglue

```sh
mkdir -p data
cd data
git clone https://github.com/timoschick/fewglue.git
cd fewglue
rm -rf .git
rm README.md
mv FewGLUE/* .
rm -r FewGLUE
cd ../..
```

Set Env variables

```sh
%env PET_ELECTRA_ROOT= /content/infoADAPET
%env PYTHONPATH=$PET_ELECTRA_ROOT:$PYTHONPATH
%env PYTHON_EXEC=python
```

Training

```sh
!python -m src.train -c /content/ADAPET/config/{config_file}.json
```

Evaluation

```sh
!python -m src.dev -e /content/ADAPET/exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/
!python -m src.test -e /content/ADAPET/exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/
```

## Model ##

<img src="img/ADAPET_update.png" width="400" height="300"/> <img src="img/LCMLM_update.png" width="420" height="300"/>
                       **Decoupled Label Loss                                                Label Conditioned Masked Language Modelling**

## Setup ##

Setup environment by running `source bin/init.sh`. This will 

- Download the [FewGLUE](https://github.com/timoschick/fewglue) and [SuperGLUE](https://super.gluebenchmark.com/tasks) datasets in `data/fewglue/{task}` and `data/superglue/{task}` respectively. 
- Install and setup environment with correct dependencies.

## Training ##

First, create a config JSON file with the necessary hyperparameters. For reference, please see `config/BoolQ.json`.

Then, to train the model, run the following commands:
```
sh bin/setup.sh
sh bin/train.sh {config_file}
```

The output will be in the experiment directory `exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/`. Once the model has been trained, the following files can be found in the directory:
```
exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/
    |
    |__ best_model.pt
    |__ dev_scores.json
    |__ config.json
    |__ dev_logits.npy
    |__ src
```

To aid reproducibility, we provide the JSON files to replicate the paper's results at `config/{task_name}.json`.

## Evaluation ## 

To evaluate the model on the SuperGLUE dev set, run the following command:
```
sh bin/dev.sh exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/
```
The dev scores can be found in `exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/dev_scores.json`.


To evaluate the model on the SuperGLUE test set, run the following command.
```
sh bin/test.sh exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/
```
The generated predictions can be found in `exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/test.json`.

## Fine-tuned Models ##

Our fine-tuned models can be found in this [link](https://drive.google.com/drive/folders/1pdVHI1Z6eRGs8OMklDCwXkPFK731yc_P?usp=sharing).

To evaluate these fine-tuned models for different tasks, run the following command:
```
python src/run_pretrained.py -m {finetuned_model_dir}/{task_name} -c config/{task_name}.json -k pattern={best_pattern_for_task}
```
The scores can be found in `exp_out/fewglue/{task_name}/albert-xxlarge-v2/{timestamp}/dev_scores.json`.
**Note**: The `best_pattern_for_task` can be found in Table 4 of the paper.

## Contact ##

For any doubts or questions regarding the work, please contact Derek ([dtredsox@cs.unc.edu](mailto:dtredsox+adapet@cs.unc.edu)) or Rakesh ([rrmenon@cs.unc.edu](mailto:rrmenon+adapet@cs.unc.edu)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

## Citation ##

Please cite us if ADAPET is useful in your work:

```
@article={tam2021improving,
          title={Improving and Simplifying Pattern Exploiting Training},
          author={Tam, Derek and Menon, Rakesh R and Bansal, Mohit and Srivastava, Shashank and Raffel, Colin},
          journal={arxiv preprint arXiv:2103.11955},
          year={2021}
}
```
