# Method2

<img src="figs/framework.png" width="600px">

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python main.py --data_dir <your data path> --task <choose genmol or gentext>
```

Set task weight in _yamls/Pretrain_Molfrag.yaml

## Evaluation

To evaluate model, run:

```eval
python eval.py --resume_from_checkpoint mymodel.ckpt 
```

## Data Preprocessing

See data_prepro/

## Core Code

See models/molfrag.py (class Molfrag)
