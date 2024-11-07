# CFASL: Composite Factor-Aligned Symmetry Learning for Disentanglement in Variational AutoEncoder

This repository is the official implementation of [CFASL: Composite Factor-Aligned Symmetry Learning for Disentanglement in Variational AutoEncoder (TMLR)](https://openreview.net/forum?id=mDGvrH7lju)
![ex_screenshot](./figs/overview_01.jpg)

## Abstract

Symmetries of input and latent vectors have provided valuable insights for disentanglement
learning in VAEs. However, only a few works were proposed as an unsupervised method,
and even these works require known factor information in the training data. We propose a
novel method, Composite Factor-Aligned Symmetry Learning (CFASL), which is integrated
into VAEs for learning symmetry-based disentanglement in unsupervised learning without
any knowledge of the dataset factor information. CFASL incorporates three novel features
for learning symmetry-based disentanglement: 1) Injecting inductive bias to align latent
vector dimensions to factor-aligned symmetries within an explicit learnable symmetry codebook 2) Learning a composite symmetry to express unknown factors change between two
random samples by learning factor-aligned symmetries within the codebook 3) Inducing a
group equivariant encoder and decoder in training VAEs with the two conditions. In addition,
we propose an extended evaluation metric for multi-factor changes in comparison
to disentanglement evaluation in VAEs. In quantitative and in-depth qualitative analysis,
CFASL demonstrates a significant improvement of disentanglement in single-factor change,
and multi-factor change conditions compared to state-of-the-art methods.

## General Settings
We set the below settings for all experiments in

**3D Cars & smallNORB**

* GPUs: NVIDIA GeForce RTX 2080 Ti x 1

**dSprites & 3D Shapes & CelebA**

* GPUs: NVIDIA GeForce RTX 3090 x 1

## Requirements

To create the environment:
    
    # cfasl.yaml file is in setup folder
    conda env create -f $DIR$/cfasl.yaml
    

#### Folders and Files
    .
    |--- configs
    |   |--- config.py                       # Model Configures (CFASL)
    |   |--- utils.py                        # Model Configures (baeslines)
    |
    |--- dataset
    |   |--- car.py                          # 3D Cars dataset loader
    |   |--- celebA.py                       # CelebA dataset loader
    |   |--- dsprites.py                     # dSprites dataset loader
    |   |--- shapes3d.py                     # shapes3d dataset loader
    |   |--- smallnorb.py                    # smallNORB dataset loader
    |   |--- utils.py                        # dataset loader
    |
    |--- model
    |   |--- betatvae.py                     # beta-TCVAE model
    |   |--- betavae.py                      # beta-VAE model
    |   |--- commutativevae.py               # Commutative Lie Group VAE model
    |   |--- controlvae.py                   # Control-VAE model
    |   |--- decoder                         # VAE decoders
    |   |--- encoder.py                      # VAE encoders
    |   |--- group_action_layer.py           # CFASL layer
    |   |--- groupbetatcvae.py               # beta-TCVAE model (CFASL)
    |   |--- groupbetavae.py                 # beta-VAE model (CFASL)
    |   |--- groupcommutativevae.py          # Commutative Lie Group VAE model (CFASL)
    |   |--- groupcontrolvae.py              # Control-VAE model (CFASL)
    |   |--- utils.py  
    | 
    |--- src
    |   |---3dplots
    |   |   |---plots.ipynb                  # Run 3D-polts from pickle files
    |   |
    |   |--- analysis_tools
    |   |   |--- common_quali.py             # -2 +2 qualitative analysis
    |   |   |--- eigen.py                    # extract eigen vector and value of latent vector space
    |   |   |--- largest_kld.py              # Lagest KLD Dimension Change (Figure 5-a)
    |   |   |--- plots.py                    # 3D-plots (Figure 1)
    |   |   |--- symmetries.py               # Other Qualitative Analysis (Figure 5-a)
    |   |   |--- utils.py
    |   |
    |   |--- disent_metrics                  
    |   |   |--- betavae.py                  
    |   |   |--- dci.py 
    |   |   |--- eval.py 
    |   |   |--- fvm.py 
    |   |   |--- mfvm.py                     # proposed metric
    |   |   |--- mig.py 
    |   |   |--- sap.py 
    |   |   |--- utils.py 
    |   |
    |   |--- train                  
    |   |   |--- evaluation.py               # model evaluation
    |   |   |--- training.py                 # model training
    |   |
    |   |--- constants.py
    |   |--- files.py                        # build model saving directory
    |   |--- info.py                         # wirte results to csv
    |   |--- optimizer.py   
    |   |--- seed.py 
    |   |--- utils.py             
    |
    |--- main.py                             # model run


## Datasets


 **dSprites**:
 Download dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz from [here](https://github.com/deepmind/dsprites-dataset).
 
 **3D Shapes**:
 Download 3dshapes.h5 file from [here](https://github.com/deepmind/3d-shapes).
 
 **3D Cars**: 
 Download this dataset in [here](http://www.scottreed.info/), Deep Visual Analogy-Making [Data].
 
 **smallNORB** & **CelebA**:
 Download the datasets from the ./dataaset/{smallnorb.py & celebA.py}
 
 
 
## Training

Set {DATA_DIR} as

* dSprites: {data dir}/{filename}

* 3D Cars: {data dir}

* smallNORB: {data dir}

* 3D Shapes: {data dir} (we transform the h5 file to images)

# Model Training and Evaluation

    #!/bin/sh
    trap "exit" INT
    CUDA_VISIBLE_DEVICES={DEVICE_IDX} python {FILE DIR}/main.py \
    --device_idx {RDEVICE_IDX} \
    --dataset CHOOSE ONE OF THEM: {dsprites, shapes3d, car, smallnorb}\
    --data_dir {DATA_DI}$ \
    --output_dir {CHECKPOINT DIR} \
    --run_file {TENSORBOSRD RUNFILE DIR} \
    --project_name {WADNB PROJECT_NAME} \
    --model_type CHOOSE ONE OF THEM: {betavae, betatcvae, controlvae, commutativevae, cfasl_betavae, cfasl_betatcvae, cfasl_controlvae, cfasl_commutativevae} \
    --latent_dim {6 (3D Shapes), 10 (Others)} \
    --split 0.0 \
    --per_gpu_train_batch_size 64 \
    --test_batch_size 64 \
    --num_epoch 0 \
    --max_steps {1,000,000 (CelebA), 500,000 (3D Shapes), 300,000 (Others)} \
    --save_steps {SAVE_STEPS} \ # set as large neumber {1e+9}
    --patience {PATIENCE} \ # set as large neumber {1e+9}
    --optimizer adam \
    --seed {1,2,3,4,5,6,7,8,9,10} \
    --lr_rate 1e-4 \
    --weight_decay 0.0 \
    --alpha 1.0\
    --gamma 1.0 \
    --lamb 1.0 \
    --quali_sampling {EQUAL TO latent_dim} \
    --do_mfvm --do_train --do_eval --write 
    Common Hyper-Parameter Settings
    --sub_sec {16 (3D Shapes), 10 (Others) } \
    --epsilon {0.1 0.01} \
    --th {0.2 0.5} \
    IF BETA-VAE or -TCVAE
    --beta {1.0 2.0 4.0 6.0} \
    IF Control-VAE
    --c {10.0 12.0 14.0 16.0}
    IF Commutative Lie Group
    --rec {0.1 0.2 0.5 0.7}


## Contributing

All content in this repository is licensed under the MIT license.

### Reference

    @article{
    anonymous2024cfasl,
    title={{CFASL}: Composite Factor-Aligned Symmetry Learning for Disentanglement in Variational AutoEncoder},
    author={Anonymous},
    journal={Submitted to Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=mDGvrH7lju},
    note={Under review}
    }