# MindCon2023_LLM_Task4-3_LlaMA



## 基于MindSpore的LlaMA模型微调

数据集准备

**School Math 0.25M Dataset**

包含约25万条由BELLE项目生成的中文数学题数据, 包含解题过程。 注意: 此数据集是由ChatGPT产生的, 未经过严格校验, 题目或解题过程可能包含错误。使用过程中请注意这一点
```
{
  "instruction": "题目：小华手里有一个装满糖果的袋子，共有12个，小明想知道里面有多少颗糖果，于是他问小华：“你手里的糖果袋子里有偶数个糖果吗？”小华回答：“有，而且多于10颗。”请问小华手里的糖果袋子里最少有几颗糖果？",
  "input": "",
  "output": "\n由题目可知：小华手里的糖果袋子里有偶数个糖果；\n又知道小华手里的糖果袋子里有多于10颗糖果。\n因为糖果数为偶数，多于10颗，所以糖果数最小必须是12。\n所以小华手里的糖果袋子里最少有12颗糖果。"
}
```
**下载数据**
```
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```


执行converter.py，使用fastchat工具添加prompts模板，将原始数据集转换为多轮对话格式。


```
!python converter.py --data_path  ./data/school_math_0.25M.json --output_path  school_math-data-conversation.json 
```
转换后格式样例：
```
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 题目：小华手里有一个装满糖果的袋子，共有12个，小明想知道里面有多少颗糖果，于是他问小华：“你手里的糖果袋子里有偶数个糖果吗？”小华回答：“有，而且多于10颗。”请问小华手里的糖果袋子里最少有几颗糖果？"}, 
{"from": "assistant", "value": "\n由题目可知：小华手里的糖果袋子里有偶数个糖果；\n又知道小华手里的糖果袋子里有多于10颗糖果。\n因为糖果数为偶数，多于10颗，所以糖果数最小必须是12。\n所以小华手里的糖果袋子里最少有12颗糖果。"}]}
```
执行preprocess.py，进行数据预处理、Mindrecord数据生成，将带有prompt模板的数据转换为mindrecord格式。
```
# 脚本路径：preprocess.py 
!python preprocess.py --dataset_type  qa --input_glob  school_math-data-conversation.json --model_file  ./checkpoint_download/llama/tokenizer.model --seq_length  2048 --output_file  belleMath2048.mindrecord 
```
## lora微调

lora微调支持使用高阶接口启动单卡微调任务
修改run_llama_7b—lora.yaml 中训练数据集路径为微调数据集路径，并在input_columns中添加labels。
将任务配置文件 run_llama_7b_lora.yaml 中的 ==== dataset config ==== 部分替换成：
```
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids", "labels"]  # "input_ids", "labels" , labels are used in instruction finetune.
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 2
  repeat: 1
  numa_enable: False
  prefetch_size: 1

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset
# if True, do evaluate during the training process. if false, do nothing.
# note that the task trainer should support _evaluate_in_training function.
do_eval: False

# eval dataset
eval_dataset: &eval_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: False
  input_columns: ["input_ids", "labels"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: False
  repeat: 1
  numa_enable: False
  prefetch_size: 1
eval_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *eval_dataset
```
修改训练时学习率和优化器参数，与预训练不同，微调学习率配置如下：
```
# optimizer
optimizer:
  type: FP32StateAdamWeightDecay
  beta1: 0.9
  beta2: 0.999
  eps: 1.e-8
  learning_rate: 1.e-5

# lr sechdule
lr_schedule:
  type: CosineWithWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0.03
  total_steps: -1 # -1 means it will load the total steps of the dataset
```
```
import  mindspore ;  
mindspore . set_context ( mode = 0 ,  device_id = 0 ) 
from  mindformers.trainer  import  Trainer 
# Initialize the pre-training task 
trainer  =  Trainer ( task = 'text_generation' , 
                  model = 'llama_7b' , 
                  args = 'run_llama_6b_lora.yaml',
                  pet_method = 'lora' , 
                  train_dataset = "belleMath2048.mindrecord" ) 
#  Call the finetune interface to fine-tune 
trainer . finetune ( finetune_checkpoint = "./checkpoint_download/llama/llama_7b.ckpt" )
```
## 推理
```

import os
import argparse
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net

from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool, get_real_rank
def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


model_type='llama_7b'
use_parallel=False
device_id=0
checkpoint_path="./output/checkpoint/rank_0/llama_7b_lora_rank_0-50_2.ckpt"
use_past=True
# 初始化单卡/多卡环境
context_init(use_parallel, device_id)

# set model config
model_config = AutoConfig.from_pretrained(model_type)
model_config.use_past = use_past

model_config.parallel_config.data_parallel = 1
model_config.parallel_config.model_parallel = 1
model_config.checkpoint_name_or_path = checkpoint_path
print(f"config is: {model_config}")

# build tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type)
# build model from config
network = AutoModel.from_config(model_config)
```

```
# 多batch输入
inputs = ["I love Beijing, because",
          "LLaMA is a",
          "Huawei is a company that"]

text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)
outputs = text_generation_pipeline(inputs)
for output in outputs:
    print(output)
```


{'text_generation_text': ['I love Beijing, because it is a city that is constantly changing. It is a city that is constantly evolving. It is a city that is constantly growing. It is a city that is constantly developing.']}
{'text_generation_text': ['LLaMA is a non-profit organization that provides a safe and supportive environment for people with mental illness to live, work, and participate in the community.']}
{'text_generation_text': ['Huawei is a company that has been around for a long time. They are a Chinese company that has been making phones for a long time. They are one of the biggest phone manufacturers in the world. They have a lot of different phones that are available for purchase.\nHuawei is a Chinese company that makes smartphones and other electronic devices.']}