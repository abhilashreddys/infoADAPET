# Enhancing Tabular Reasoning with Pattern Exploiting Training #

This repository contains the modified official code for the paper: "[Enhancing Tabular Reasoning with Pattern Exploiting Training](https://vgupta123.github.io/docs/pettable.pdf)". Visit the project page at [https://infoadapet.github.io/](https://infoadapet.github.io/)

[ACL 2020](https://acl2020.org/) paper: [INFOTABS: Inference on Tables as Semi-structured Data](https://www.aclweb.org/anthology/2020.acl-main.210.pdf). To explore the dataset online visit [project page](https://infotabs.github.io).

[Infotabs Dataset](https://github.com/infotabs/infotabs)

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
Download data (if required) \
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

## Contact ##

For any doubts or questions regarding the work, please contact Abhilash ([sareddy53@gmail.com](mailto:sareddy53@gmail.com)). For any bug or issues with the code, feel free to open a GitHub issue or pull request.

