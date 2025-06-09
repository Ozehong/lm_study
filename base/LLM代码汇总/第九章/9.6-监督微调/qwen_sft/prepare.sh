# 下载微调训练数据集
modelscope download --dataset 'BAAI/Infinity-Instruct' --local_dir 'data/sft' # 选择7M和Gen进行微调，因为这两个数据集更新时间最近，且数据量大
