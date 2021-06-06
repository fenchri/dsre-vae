wget https://github.com/ZhixiuYe/Intra-Bag-and-Inter-Bag-Attentions/raw/master/NYT_data/NYT_data.zip
unzip NYT_data.zip

python fix_format.py train.txt nyt10_570k_train_full.txt
cd ../

python clean_n_split.py --input_data nyt10_570k/nyt10_570k_train_full.txt \
                        --r2id nyt10/nyt10_rel2id.json \
                        --train_data nyt10_570k/nyt10_570k_train.txt \
                        --val_data nyt10_570k/nyt10_570k_val.txt \

