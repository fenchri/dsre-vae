#!/usr/bin/env bash

for dataset in 'Freebase' 'Wikidata';
do
    for embed_dim in 64;
    do
        mkdir -p "./${dataset}/ckpts_${embed_dim}/"

        dglke_train --model_name TransE_l2 \
                    --dataset ${dataset} \
                    --data_path ./${dataset} \
                    --format raw_udd_hrt \
                    --data_files train.tsv valid.tsv test.tsv \
                    --batch_size 1024 \
                    --neg_sample_size 256 \
                    --hidden_dim ${embed_dim} \
                    --gamma 10 \
                    --lr 0.1 \
                    --regularization_coef 1e-7 \
                    --batch_size_eval 1000 \
                    --test -adv \
                    --gpu 0 \
                    --max_step 500000 \
                    --neg_sample_size_eval 1000 \
                    --log_interval 1000 \
                    --save_path "./${dataset}/ckpts_${embed_dim}/"
    done
done