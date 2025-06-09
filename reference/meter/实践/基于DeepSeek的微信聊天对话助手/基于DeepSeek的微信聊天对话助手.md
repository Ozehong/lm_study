&#x20;  &#x20;

> # 时间不在于你拥有多少，而在于你怎样使用。
>
> 别等到日子像水一样流走，才发现自己一直站在原地。

> **简介：你好呀👋，欢迎来到我打造的Meteor导航站。这里记录了我从零开始学习大模型的全过程，包括学习路线图、基础原理、项目实战、面试题库等内容。如果你也在探索大模型的世界，希望这份笔记能给你一些启发 🌟**
>
> **这个项目能够帮助你快速训练一个属于自己的模型，并将预训练、后训练、微调、dpo流程打通。**

***

## 基于DeepSeek的女朋友专属助手（用和女朋友的聊天记录微调DeepSeek）

代码已打包放在这里了。

[llm-deepseek.zip](files/llm-deepseek.zip)

### 0. 项目详细指南

目前正在开发ing，后续出教程，这里简单记录一下，最后在整理

第一步：先确定玩的模型

* 由于deepseek系列的模型没有比较小的聊天模型，但是大一点的又玩不起，只能硬着头皮去用DeepSeek-R1-Distill-Qwen-1.5B去做了。https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

第二步：确定数据集

* 预训练数据集：找了一个中文的预训练数据集，也是在文档的常用表格里又，哈哈有笔记真的好方便！https://huggingface.co/datasets/Skywork/SkyPile-150B

* 微调数据集：当然是自己的微信聊天记录了\~

第三步：提取和女朋友的微调数据并转成json

* 使用留痕软件提取：https://memotrace.cn/

* WeChatMsg：https://github.com/LC044/WeChatMsg

> 🎉这个代码把路径配kk数解析类
>
> * 加入lora进行微调

### 1. 预训练DeepSeek（Hugging Face简单版）

```python
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# 定义路径
class PreTrainArguments:
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    train_data_path = "/openbayes/home/opentest/datasets/Sky_train.jsonl"
    test_data_path = "/openbayes/home/opentest/datasets/Sky_test.jsonl"

args = PreTrainArguments()

# 定义tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# 加载数据集
# streaming是有惰性的，不会立即执行，只有在需要的时候才会执行
data_files = {
    "train": args.train_data_path,
    "test": args.test_data_path
}
dataset = load_dataset("json",data_files=data_files)
print(dataset)

def tokenize_to_id(samples):
    batch_text = samples["text"]
    outputs = tokenizer(batch_text, truncation=True, max_length=512, return_attention_mask=False)
    input_ids = [np.array(item) for item in outputs["input_ids"]]
    return {"input_ids": input_ids}
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset = train_dataset.map(tokenize_to_id, batched=True)
test_dataset = test_dataset.map(tokenize_to_id, batched=True)
train_dataset = train_dataset.select(range(10000))
print(train_dataset)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./results',            # 保存模型的目录
    num_train_epochs=3,                # 训练轮数
    per_device_train_batch_size=1,     # 训练批次大小
    per_device_eval_batch_size=1,      # 验证批次大小
    warmup_steps=500,                  # 训练的预热步数
    weight_decay=0.01,                 # 权重衰减
    logging_dir='./logs',              # 日志目录
    logging_steps=10,                  # 多少步记录一次日志
    evaluation_strategy='steps',       # 评估策略
    save_strategy='epoch',             # 保存策略
    learning_rate=1e-4,                # 学习率
    eval_steps=1000000,                # 多少步评估一次
    report_to="wandb"                  # 看loss曲线
)

# 定义模型
deepseek = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B")
# 定义LoRA的配置
lora_config = LoraConfig(
    r=16,                       # LoRA秩 (越大表示保留更多信息)
    lora_alpha=32,             # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 指定哪些层使用LoRA
    lora_dropout=0.05,         # Dropout以防止过拟合
    bias="none",               # 不更新偏置
    task_type="CAUSAL_LM"      # 指定任务类型
)

deepseek = get_peft_model(deepseek, lora_config)
deepseek.print_trainable_parameters()

# 开始训练!
trainer = Trainer(
    model=deepseek,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 2. 微调DeepSeek（Hugging Face简单版）

> 目前仅将html中的数据提取到了json格式，明天在更完整版数据集。

#### 2.1 提取数据集

```python
from bs4 import BeautifulSoup
import json

# 读取 HTML 文件内容
file_path = 'chat.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# 解析 HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 查找所有 <script> 标签
script_tags = soup.find_all('script')

messages = []

# 遍历所有 <script> 标签并查找包含 'chatMessages' 的脚本
for script_tag in script_tags:
    if script_tag.string and 'chatMessages' in script_tag.string:
        content = script_tag.string
        # 提取 chatMessages 数组的 JSON 字符串
        start_index = content.find('chatMessages = ') + len('chatMessages = ')
        end_index = content.find('];', start_index) + 1
        chat_messages_str = content[start_index:end_index]

        try:
            # 解析 JSON
            chat_messages = json.loads(chat_messages_str)

            for item in chat_messages:
                if 'text' in item:
                    role = 'assistant' if item['is_send'] == 1 else 'user'
                    messages.append({"role": role, "content": item['text']})

            # 处理合并连续同角色的消息
            merged_messages = []
            for msg in messages:
                if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                    merged_messages[-1]["content"] += " " + msg["content"]
                else:
                    merged_messages.append(msg)

            formatted_conversations = []
            formatted_conversation = []
            for i in range(len(merged_messages)-1):
                formatted_conversation.append({"role": merged_messages[i]["role"], "content": merged_messages[i]["content"]})
                if len(formatted_conversation) == 2:
                    formatted_conversations.append(formatted_conversation) 
                    formatted_conversation = []

            # 保存到 JSON 文件
            with open("chat_dataset.json", "w", encoding="utf-8") as f:
                json.dump(formatted_conversations, f, ensure_ascii=False, indent=4)

            print("✅ 处理完成！合并后的数据已保存到 chat_dataset.json")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析错误: {e}")

        break  # 找到 chatMessages 后退出循环
```

#### 2.2 SFT训练

```python
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer
from peft import LoraConfig, get_peft_model


class SFTArguments():
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    train_data_path = "/openbayes/home/opentest/data/chat_dataset.json"
    pretrain_lora_path = "/openbayes/home/opentest/deepseek/results/checkpoint-30000/adapter_model.safetensors"


args = SFTArguments()

# ✅ 修正 train_dataset 取 "train" 数据集
train_dataset = load_dataset("json", data_files=args.train_data_path)
print(train_dataset)
# ✅ 修正 tokenizer 使用方式
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize_to_id(example: dict) -> dict:
    messages = [
        {"role": "user", "content": example["boy"]},
        {"role": "assistant", "content": example["girl"]}
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False
    )

    # Tokenize完整对话
    input_ids = tokenizer(
        formatted_text,
        truncation=True,
        max_length=1024,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]

    # 查找助理回复的起始位置
    assistant_index = formatted_text.find("<｜Assistant｜>")
    prefix_text = formatted_text[:assistant_index]
    seq_len = len(tokenizer(prefix_text, add_special_tokens=False, max_length=1024, truncation=True)["input_ids"])
    return {"input_ids": input_ids, "seq_len":seq_len}


# ✅ 修正 train_dataset.map
train_dataset = train_dataset["train"]
train_dataset = train_dataset.map(tokenize_to_id, remove_columns=["boy","girl"])
train_dataset = train_dataset.select(range(10000))

def data_collator(features):
    len_ids = [len(feature["input_ids"]) for feature in features]
    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]

        labels = [-100] * seq_len + ids[seq_len:]

        labels_list.append(torch.LongTensor(labels))
        input_ids.append(torch.LongTensor(ids))
        if len(labels) != len(ids):
            print()
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels_list)}


training_args = TrainingArguments(
    output_dir='./sft_test_results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./sft_logs',
    logging_steps=10,
    evaluation_strategy='steps',
    save_strategy='epoch',
    learning_rate=5e-5,
    eval_steps=1000000,
)

deepseek = AutoModelForCausalLM.from_pretrained(args.model_name)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
deepseek = get_peft_model(deepseek, lora_config)

# ✅ 修正 load_state_dict
state_dict = load_file(args.pretrain_lora_path)
missing_keys, unexpected_keys = deepseek.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

deepseek.print_trainable_parameters()

trainer = Trainer(
    model=deepseek,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### 3. DPO强化DeepSeek

#### 3.1 DPO数据集

需要制作一下数据集，我们可以用SFT后的模型对测试集进行评估，将模型生成的结果存储为rejected，然后将标准答案存储为chosen

#### 3.2 DPO训练

```python
import sys
from dataclasses import dataclass
from typing import Dict

from datasets import load_dataset, Dataset
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig,DPOTrainer

def get_dataset(file) -> Dataset:
    dataset = load_dataset('json', split="train", data_files=file)
    def split_prompt_responses(sample) -> Dict[str, str]:
        return {
            "prompt": f"{sample['boy']}<｜end▁of▁sentence｜>",
            "chosen": f"{sample['boy']}<｜end▁of▁sentence｜>",
            "rejected": f"{sample['girl']}<｜end▁of▁sentence｜>"
        }
    return dataset.map(function=split_prompt_responses).shuffle(12345)

def train_dpo(config, peft_config):
    model_name = "/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B"
    sft_lora_path = "/openbayes/home/opentest/deepseek/sft_test_results/checkpoint-30000/adapter_model.safetensors"
    # step 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # step 2. 加载SFT模型
    model_train = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = load_file(sft_lora_path)
    model_train.load_state_dict(state_dict, strict=False)

    model_ref = AutoModelForCausalLM.from_pretrained(model_name)
    state_dict = load_file(sft_lora_path)
    model_ref.load_state_dict(state_dict, strict=False)

    train_dataset = get_dataset("/openbayes/home/opentest/data/train_chat_dataset.json")
    eval_dataset = get_dataset( "/openbayes/home/opentest/data/train_chat_dataset.json")

    dpo_trainer = DPOTrainer(
        model_train,
        model_ref,
        peft_config=peft_config,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
        # data_collator=data_collator
    )

    dpo_trainer.train()
    dpo_trainer.save_model(config.model_save_dir)



if __name__ == '__main__':

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # text 2 text lora model
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
    )

    dpo_config = DPOConfig(
        output_dir='./sft_test_results',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./sft_logs',
        logging_steps=10,
        evaluation_strategy='steps',
        save_strategy='epoch',
        learning_rate=5e-5,
        eval_steps=1000000,
        force_use_ref_model=True
    )

    train_dpo(dpo_config, peft_config)
```

### 4. Hugging face实现 -> torch实现

#### 4.1 数据预处理

```python
import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="/openbayes/home/opentest/datasets/Sky_train.jsonl" ,required=False, help='Directory including the configuration file')
    return parser.parse_args()

def load_json_to_token(file):
    train_data_name = "/openbayes/home/opentest/datasets/pre_train.pt"
    eval_data_name = "/openbayes/home/opentest/datasets/pre_test.pt"
    with open(file, 'r', encoding='utf-8') as f:
        examples_list = json.load(f)
    result = []
    for examples in tqdm(examples_list, desc="Processing examples"):
        input = tokenizer(examples["boy"], truncation=True, max_length=512, return_attention_mask=False)["input_ids"]
        label = input
        input = torch.LongTensor(input)
        label = torch.LongTensor(label)
        result.append({"input_ids": input,"label": label})
    torch.save(result[1000:], train_data_name)
    torch.save(result[1000:], eval_data_name)
    return torch.load(train_data_name), torch.load(eval_data_name)




if __name__ == '__main__':
    args = parse_args()
    train_file = "/openbayes/home/opentest/data/train_chat_dataset.json"
    load_json_to_token(train_file)
```

#### 4.2 数据加载

```python
from torch.utils.data import DataLoader, Dataset, IterableDataset
class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

#### 4.3 手动实现训练

```python
from transformers import AutoModelForCausalLM
import torch
from process_data import load_json_to_token
from dataset import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
# ==========================处理数据========================================
train_file = "/openbayes/home/opentest/data/train_chat_dataset.json"
# 1. 处理数据
train_data, test_data = load_json_to_token(train_file)
# 2. 加载数据
dataset = MyDataSet(train_data)
# 3. 将加载的数据转为可迭代的批量数据加载器
dataloader = DataLoader(dataset, batch_size=1)

# ==========================加载模型=======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 加载模型
deepseek = AutoModelForCausalLM.from_pretrained("/openbayes/input/input0/DeepSeek-R1-Distill-Qwen-1.5B").to(device)
# 1.1 全参数训练用显存太大了，只是为了学习测试，只训练其lm_head
for name, param in deepseek.named_parameters():
    print(name, param.shape)
    if name == "lm_head.weight":
        continue
    param.requires_grad = False
# 2. 定义优化器
optimizer = torch.optim.AdamW(deepseek.parameters(), lr=0.0001, betas=(0.9, 0.95), weight_decay=0.1)
# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()
# ========================开始训练=======================================
for epoch in range(1):
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        deepseek.train()
        input_ids, label = batch["input_ids"].to(device), batch["label"].to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = deepseek(input_ids)  # 前向传播
        shift_logits = outputs.logits[:, :-1, :].contiguous()  # 形状 [batch_size, seq_len-1, vocab_size]
        shift_labels = label[:, 1:].contiguous()  # 形状 [batch_size, seq_len-1]
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()
```

### 项目中遇到的问题

> 项目遇到的问题：
>
> 1. 没有label报错，用DataCollatorForLanguageModeling自动生成label
>
> 2. 看loss曲线方法：使用tensorborad：ssh -L 本地端口:127.0.0.1:TensorBoard端口 用户名@服务器的IP地址 -p 服务器登录端口 -> 改用wandb
>
> 3. 训练效果不是很好，怀疑自己代码问题，所以读了 hugging face 的源码，懂了更多，以及如何制造label，各个部分的函数是什么作用以及底层原理的实现
>
> 4. 对Post-Train、SFT、DPO有了一定了解，但是RLHF的知识以及DPO还需加强
>
> 阅读了hf的一些源码，收获较多。虽然用的hf方式实现较为简单，但是后续会做一遍自己造轮子去实现，也正好是下一个项目的开始：pip install meteor拥有一个属于自己的包，后续直播带大家做

