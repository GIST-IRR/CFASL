#!/bin/sh
trap "exit" INT

CUDA_VISIBLE_DEVICES=0 python /home/s1_u1/projects/CFASL/main.py \
--device_idx 0 \
--dataset dsprites \
--data_dir /home/s1_u1/datasets/disentanglement/2D_shapes/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
--output_dir /home/s1_u1/checkpoints/Group_VAE_sub/2Dshape \
--run_file /home/s1_u1/runs/Group_VAE_sub/2Dshape \
--project_name cars_group_equiv_attn_mean_std_mse_betavaes_refactor33_ablation2 \
--model_type groupbetatcvae \
--latent_dim 10 \
--split 0.0 \
--per_gpu_train_batch_size 64 \
--test_batch_size 64 \
--num_epoch 0 \
--max_steps 10 \
--save_steps 100000000 \
--patience 70000000 \
--optimizer adam \
--seed 1 \
--lr_rate 1e-4 \
--weight_decay 0.0 \
--alpha 1.0 \
--gamma 1.0 \
--lamb 1.0 \
--quali_sampling 10 \
--do_mfvm --do_train --do_eval --write \
--sub_sec 20 \
--epsilon 0.1 \
--th 0.2 \
--beta 1.0