# 代码说明


## 程序手册

### runs
- 训练sar模型的sh文件
- MLE_sar_chunk_chunkposition.sh

 ~~~python
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

--fp16 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \

--lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \

--no-epoch-checkpoints \

--optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \

--skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \

--keep-interval-updates 2 --no-progress-bar --log-interval 100 \

--criterion cross_entropy_wcustom_metrics \

--save-dir $SAVE_DIR \

--tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt

  

rm -rf /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/models/.ipynb_checkpoints
 ~~~

### fairseq/models
-  transformer_sar.py
	- main function：该文件定义了sar decoder模型
	- 定义 TransformerSarDecoder 类，继承于 transformer.py 文件中TransformerDecoder类，TransformerSarDecoder 类别将会在 transformer_lm_sar.py 文件中实例化，该类编写了sar模型的主要结构
	- 重写__init__（），并且初始化两种positional embedding
	- 重写 extract_features() 方法，重新编写模型的forward部分，替换了 ar 模型的 mask 矩阵和 position encoding

- transformer_lm_sar.py
	- main funcion：该文件整合了基础类别，定义了sar语言模型
	- 定义 TransformerSarLanguageModel 类，该类继承于transformer_lm.py 文件中的 TransformerLanguageModel 类，
	- 重写 build_model() 方法，在 build_model() 中定义 TransformerSarDecoder 
	- 注册 transformer_sar_lm 模型
	- 注册  transformer_sar_lm_big 模型

### fairseq/moduels
- interchunk_learned_positional_embedding.py
	- main function：该类主要定义了chunk之间learned positional embedding
	- 定义InterchunkLearnedPositionalEmbedding类，继承于 nn.Embedding, 该类会在transformer_sar.py 文件中实例化
- insidechunk_learned_positional_embedding.py
	- main function：定义chunk内部的positional embedding
	- 定义 InsidechunkLearnedPositionalEmbedding 类，继承于 nn.Embedding, 该类同样会在transformer_sar.py 文件中实例化

### fairseq/data
- add_chunkstamp_dataset.py
	- main function：针对 ar 模型训练 chunked wikitext 数据集所编写的 dataloader 函数。wikitext-103经过spacy 处理过后的数据chunked wikitext103，只有属于 chunk 的数据加入了<chunk_s> 与<chunk_e>特征，非chunk的数据并没有加入<chunk_s> 与<chunk_e>特征。但是模型在训练时将非chunk也加入 <chunk_s> 与<chunk_e>特征，认为其同样是一个chunk
	- 定义class AddChunkStampDataset 继承于 MonolingualDataset，AddChunkStampDataset 类将在 language_modeling_with_generation_ar_chunk.py 中进行实例化

- chunked_dataset.py
	- main function：针对sar模型训练 chunked wikitext 数据集所编写的 dataload 函数。wikitext-103经过spacy 处理过后的数据chunked wikitext-103，只有属于 chunk 的数据加入了<chunk_s> 与<chunk_e>特征，非chunk的数据并没有加入<chunk_s> 与<chunk_e>特征。但是sar模型在训练时将非chunk也加入 <chunk_s> 与<chunk_e>特征，认为其同样是一个chunk
	- 并且该dataloader 实现对过长的chunk进行截断的功能
	- 定义class ChunkedDataset， 继承于 MonolingualDataset，ChunkedDataset 在 language_modeling_with_generation_sar_chunk.py 中进行实例化


### fairseq/customs
- evaluation_chunked_data.py
	- 编写专门用于评价文本中带有chunk stamps（<chunk_s> <shunk_e>）的数据，当前版本代码在 evaluate_utils.generate_completions() 模型生成tokens之后删除全部的 <chunk_s> <shunk_e> 是的数据中没有chunk 特征，进行正常的mauve的计算

- language_modeling_with_generation_ar_chunk.py
	- 注册'language_modeling_with_generation_ar_chunk' 任务，用于 ar 模型训练 chunked wikitext 数据集
	- 实例化 AddChunkStampDataset 类别，该类为ar模型训练chunker wikitext-103 设计

- language_modeling_with_generation_sar_chunk.py
	- 注册 'language_modeling_with_generation_sar_chunk' 任务，用于 sar 模型训练 chunked wikitext 数据集
	- 实例化 ChunkedDataset 类别，该类为sar模型训练chunker wikitext-103 设计

- transformer_arch.py
	- 注册 'transformer_sar_lm_ul' 
	- 注册 'transformer_sar_lm_debug'