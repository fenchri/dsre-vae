# Download Data
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_test.txt
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_train.txt
# wget https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_rel2id.json

mv nyt10_train.txt nyt10_train_full.txt

# Split Data
cd ../
python clean_n_split.py --input_data nyt10/nyt10_train_full.txt \
                        --train_data nyt10/nyt10_train.txt \
                        --r2id nyt10/nyt10_rel2id.json \
                        --val_data nyt10/nyt10_val.txt
