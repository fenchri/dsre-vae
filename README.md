# DSRE with Knowledge Base Priors

Source code for "[Distantly Supervised Relation Extraction with Sentence Reconstruction and Knowledge Base Priors]()" in NAACL 2021

#### Reference
If you find this code useful and plan to use it, please cite the following paper =)

```
@inproceedings{christopoulou-etal-2021-distantly,
    title = "Distantly Supervised Relation Extraction with Sentence Reconstruction and Knowledge Base Priors",
    author = "Christopoulou, Fenia  and
      Miwa, Makoto  and
      Ananiadou, Sophia",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.2",
    pages = "11--26"
} 
```

## Prerequisites

### Environment
```bash
conda create -n dsre-vae python=3.7.3
conda activate dsre-vae
pip install -r requirements.txt
```
All models were trained on a single Nvidia v100 GPU.

### Pretrained word embeddings
```bash
mkdir embeds
cd embeds 
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### Datasets & Pre-processing

Download and process the datasets as follows:
```bash
# Language Data
cd data/LANG/nyt10/
sh make_data.sh
cd ..
python preprocess.py --max_sent_len 50 --lowercase --max_bag_size 500 --path nyt10/ --dataset nyt10

cd data/LANG/nyt10_570k/
sh make_data.sh
cd ..
python preprocess.py --max_sent_len 50 --lowercase --max_bag_size 500 --path nyt10_570k/ --dataset nyt10_570k

cd data/LANG/wiki_distant/
sh make_data.sh
cd ..
python preprocess.py --max_sent_len 30 --lowercase --max_bag_size 500 --path wikidistant/ --dataset wiki_distant


# KB data
cd data/KB/
python make_data.py --data Freebase \
                    --train_file ../LANG/nyt10/nyt10_train.txt \
                    --val_file ../LANG/nyt10/nyt10_val.txt \
                    --test_file ../LANG/nyt10/nyt10_test.txt 
                    
python make_data.py --data Wikidata \
                    --train_file ../LANG/wikidistant/wiki_distant_train.txt \
                    --val_file ../LANG/wikidistant/wiki_distant_val.txt \
                    --test_file ../LANG/wikidistant/wiki_distant_test.txt 

```


## Training

### Training KB embeddings
In order to train Knowledge Base embeddings, we will use the [DGL-KE](https://github.com/awslabs/dgl-ke) package.
Follow the instructions on the page for installation.
```bash
cd data/KB
sh train_embeds.sh
```
Embeddings will be saved in the `Freebase/ckpts_64/` and `Wikidata/ckpts_64/` directories, respectively.

> The Link Prediction algorithm will not produce the same embeddings each time. 
> For reproducibility, we share the KB priors that we used in the paper.
> Download them from [here](https://drive.google.com/file/d/1rqXQ3uqI0n98S5j7gPYaXgf1hQELIi_E/view?usp=sharing) and place each `pairs_mu_64d.json` file in the `data/KB/appropriate_folder` directories.

It is not necessary to run the following code, except if you want to collect priors for your own KB:
```bash
python calculate_priors.py --kg_embeds Freebase/ckpts_64/TransE_l2_Freebase_0/Freebase_TransE_l2_entity.npy \
                           --e_map Freebase/entities.tsv \
                           --data ../LANG/nyt10/nyt10_train.txt \
                           --kg Freebase
```



### Training DSRE Models
Models can be trained in three modes: baseline, reconstruction only and reconstruction with KB priors.  
At the end of training, the log file and PR curve points are saved in the `saved_models` directory.

```bash
cd src/
python main.py --config ../configs/nyt10_base.yaml --mode train   # baseline
python main.py --config ../configs/nyt10_reco.yaml --mode train   # N(0, 1) prior
python main.py --config ../configs/nyt10_prior.yaml --mode train  # N(mu, 1) prior
```

In order to just test a model, you can use the `--mode test` argument.
The `--mode infer` mode (a) produces 20 random sentences from the latent code of the VAE, (b) produces a set of homotomies, 
(c) collects the μ vectors of sentences as produced by the VAE (which is necessary for plotting the space of the posteriors).


### Plot
T-SNE plots can be obtained from running
```bash
cd src/helpers/
python plot_tsnes.py --priors ../../data/KB/Freebase/pairs_mu_64d.json \
                     --posteriors file_with_posteriors \
                     --dataset nyt10 \
                     --filename ../../data/LANG/nyt10/nyt10_val.txt \
                     --legend \
                     --split val
```

PR-curves can be download from [here](https://drive.google.com/file/d/1AEOVfoAhBNr3G1bwQ5_8V5WqmfLeyA06/view?usp=sharing). 
Place the `pr_curves/` directory inside `dsre-vae/`. 
Then run the following:
```bash
cd src/helpers/
python plot_pr.py --dataset nyt10
```


