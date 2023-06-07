rm -rf /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/models/.ipynb_checkpoints

SAVE_DIR=/ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/checkpoints/model_sar_chunk
mkdir -p $SAVE_DIR

export HOME=/ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export device='cuda'
#unset WORLD_SIZE
unset MASTER_PORT
unset RANK
unset MASTER_ADDR

python -u -W ignore /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/train.py \
--task language_modeling_with_generation_sar_chunk /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/datas/data-bin/chunked_wikitext-103 \
    --user-dir /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/custom --arch transformer_sar_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \
    --fp16  --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
    --no-epoch-checkpoints \
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \
    --keep-interval-updates 2 --no-progress-bar --log-interval 100 \
    --criterion cross_entropy_wcustom_metrics \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt

rm -rf /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/models/.ipynb_checkpoints