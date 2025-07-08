# jiutian-chat项目使用说明：

## 1. jiutian-chat 文件说明：

- `deepspeed-jiutian-chat`: 微调代码
- `models`: 模型结构，模型下载路径包括：
  - [九天汇聚平台](https://jiutian.10086.cn/qdlake/qdh-web/#/model/detail/1070)
  - [ModelScope](https://www.modelscope.cn/models/JiuTian-AI/JIUTIAN-139MoE-chat)
- `README.md`: 包含相关依赖说明
- `requirements.txt`: 列出了环境依赖



## 2. 创建环境
- Python 版本: 3.10
- Torch 版本: 2.0
- Transformers 版本: 3.36.2
- CUDA 版本: 11.7
- Deepspeed 版本: 1.4

**步骤**:
1. `conda create -n jiutian2sft python=3.10.14`
2. `pip install -r requirements.txt`
3. `conda activate jiutian2sft`

## 3. 数据处理：
1. 进入 `deepspeed-jiutian-chat/data` 目录
执行 `python generate_jiutian2_sft_data.py`
2. 修改 `generate_jiutian2_sft_data.py` 文件中的 `model_path`、`inputfile`、`out_file`
3. **数据格式**:
     ```json
     {
       "conversations": [
         {"from": "human", "value": "如何利用大数据分析提高酒店入住率？"},
         {"from": "assistant", "value": "利用大数据分析可以显著提高酒店的入住率和客户满意度。以下是一些方法，......以保障客户信息的安全。"}
       ]
     }
     {
       "conversations": [
         {"from": "human", "value": "\"你有兄弟姐妹吗？\""},
         {"from": "assistant", "value": "作为AI语言模型，我是一个程序，没有家庭和亲属，因此我没有兄弟姐妹。我的存在只是为了回答您的问题和提供信息。如果您有任何其他问题或需要帮助，请随时向我询问。"}
       ]
     }
     ```

## 4. 模型微调：
- 进入 `deepspeed-jiutian-chat` 目录
- 配置 `hostfile` 文件
- 执行 `bash run.sh` 启动微调


## 5. 模型推理示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "/jiutian-chat/modepathxxx"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
text = "介绍一下北京的故宫"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.03)
print(tokenizer.decode(outputs[0]))
```







