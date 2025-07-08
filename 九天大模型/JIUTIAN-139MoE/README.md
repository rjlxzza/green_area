# JIUTIAN-139MoE
<p align="center">
 🤖 <a href="https://www.modelscope.cn/models/JiuTian-AI/Jiutian-139MoE-chat" target="_blank">ModelScope</a>  🟣 <a href="https://jiutian.10086.cn/qdlake/qdh-web/#/model/detail/1070." target="_blank">qdlake</a> 



## Introduction

We report the development of JIUTIAN-139MoE, a 13-billion active parameter language model
designed to be an efficient foundation model for industrial use. It adopts a decoder-only Transformer-based Mixture-of-Experts (MoE) architecture, employing a pair of large twin experts and six small
experts to capture the intelligence associated with diverse industries. In terms of training, we support
training with clusters of various GPUs and NPUs. We also support lossless switch between two
heterogeneous clusters. In addition, JIUTIAN-139MoE-Chat, a fine-tuned version of JIUTIAN-
139MoE, surpasses state-of-the-art large language models on both open and self-built industrystandard benchmarks. Specifically, it exhibits outstanding performances on 10 industrial benchmarks
and leading performances on 10 benchmarks of general knowledge understanding, reasoning, math
and coding capabilities. JIUTIAN-139MoE is released under the Apache 2.0 license and [JIUTIAN Large Model Community License Agreement](./JIUTIAN_Large_Model_Community_License_Agreement.pdf). It is publicly available at https://jiutian.10086.cn/qdlake/qdh-web/#/model/detail/1070.
## Model Architecture

JIUTIAN-139MoE is based on a standard decoder-only transformer architecture, similar to LLaMA. Different from LLaMA, the modification of JIUTIAN-139MoE is mainly at the FeedForward Network (FFN)
layers, as illustrated in Figure 1. The modifications are further described below, and the configuration details of
JIUTIAN-139MoE are shown in Table 1.


### FeedForward Network
FeedForward Network. Similar to Mixtral 8x7B, the FFNs of JIUTIAN-139MoE are replaced by Mixture-of-Expert layers. Here, we adopt "eight" experts (with a pair of virtual twin experts) and a randomly initialized router as the MoE layers. In particular, we introduce a pair of twin experts and special expert activation strategy. We first train a dense model and then expand to the MoE architecture. In order to preserve the ability of the original dense model, we replace one of twin experts directly with the copy of the FFN in original dense model. Other experts are obtained by randomly partitioning the original FFN, which can reuse the sunk training costs. It further improves the performance of the model by utilizing additional experts to enhance the characterization of features while retaining the original dense model capability.

**Figure 1**

![](https://krseoul.imgtbl.com/i/2024/07/05/668790ab19e98.jpg)


**Table 1**


| Params Number | Layer Number |Hidden Size|Heads|Sequence Length|Vocabulary Size|Experts Number|FFN Hidden Size|Activated Params Number|
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |                
| 38.8B	| 40	| 5120	| 40| 	4096| 	69120	| 7	| 13824	| 13B| 





## Training Infrastructure
Most of our training work is conducted on our self-developed Jiutian Intelligent computing platform, which includes heterogeneous computing resources such as GPU and NPU. It utilizes high-speed 1.6T InfiniBand or RoCE non-blocking high-speed interconnect networks, coupled with high-performance dedicated storage. As mentioned in this paper, we use 896 GPU cards in the training stage. The platform achieves task monitoring and automatic recovery from interruptions, efficiently managing high-performance computing units and high-speed networks, thereby ensuring the stability and efficiency of the training process.



## Training Data
We collect pretraining data from various types of data sources: web-pages, books, news articles, industrial knowledge
materials, academic papers, etc. Most of the datasets are written in English and Chinese languages. We still have a
small portion of other countries’ languages (e.g., Indonesian, Spanish, Arabic, Russian) and code data in Python, C++,
or other programming languages.

We have dedicated a lot of effort to collecting industrial knowledge data. Here "industrial knowledge data" are defined
as the data that are highly related to industrial production, technology development, economic growth, environmental
protection, medical care, and so on. The important industrial areas are telecommunications, energy, transportation,
aviation, steel, finance, etc. Those data are mainly collected through 5 types of sources:

• Public academic papers, documents, books about industrial knowledge;

• Valuable documents that are filtered, cleaned, and extracted from public web page data;

• Industrial knowledge problems such as exams, exercises, Q&As;

• Structured knowledge graph data;

• Public high-quality data provided by our cooperative partners, for example, other companies or departments
from China Mobile Group.

• A comprehensive data preprocessing pipeline is developed to improve the quality of the pretraining data, including:

• Filtering the documents which are illegal, unsafe, unreadable, or advertisements;

• Cleaning the incoherent phrases or characters such as emojis, HTML tags &links, etc;

• Exact and near deduplication.

It is critical to remove illegal or unsafe documents from the pretraining data. Here ’illegal’ or ’unsafe’ means the content
of the document has issues about politics, race, privacy, violence, etc. We develop model-based and heuristic methods
to conduct the detection and filtering at document, sentence, and even word level. As model-based methods, we collect
different types of unsafe documents and sentences, together with normal ones, to train a classifier as the detector. As
heuristics, we develop a lot of scripts, patterns, and word lists to locate the abnormal content at a phrase or word level.

Apart from illegal and unsafe documents, there are still some low-quality documents such as noisy or unreadable
paragraphs, advertisements, HTML tags, java scripts, and emojis. We clean those types of documents using both
model-based and heuristic methods as well.

There are two types of duplications: exact duplicates and near duplicates. The simple document matching is used to
remove the exact duplicates. And the Minhash LSH deduplication [Broder, 1997] is used to remove the near duplicates.
Table 2 shows the retention rate after deduplication on our English and Chinese datasets. Since the web-page-related
dataset, e.g., common-crawl [Radford et al., 2019a], has the most duplicates, their retention rates are listed separately.
After filtering, cleaning, deduplication and tokenization, finally we built a dataset encompassing 5 trillion tokens for the
pretraining process.



**Table 2**

| Language | All Datasets | Web-page Datasets |
|----------|--------------|-------------------|
| English  | 70.1%        | 59.3%             |
| Chinese  | 73.0%        | 69.9%             |




## Tokenization

In our work, we utilize byte pair encoding (BPE) [Sennrich et al., 2015] implemented in the SentencePiece framework
[Kudo and Richardson, 2018] as our tokenization method. To enhance the performance of our model on multilingual
 downstream tasks and downstream tasks in specialized industrial sectors, we train the tokenizer on a smaller subset of
the training corpus as described in Subsection 2.3. To ensure computational efficiency during training and to reserve
space for any additional special tokens that might be needed in the future, we augment the final vocabulary with 553
special tokens and configure the final model’s vocabulary size to 69,120 for training.


## Alignment
### Data Preprocessing

To fully unlock the potential of pre-trained models in chat use cases and better align the model with human values
and expectations, it is necessary to perform post-training via Supervised Fine-Tuning (SFT) [Ouyang et al., 2022],

Direct Preference Optimization (DPO) [Rafailov et al., 2024] etc. The curation of a high-quality training dataset is
paramount. We have constructed a dataset containing tens of millions of instruction data instances and implemented a
detailed hierarchical system for this dataset, which includes 113 domains and 53 capabilities. As illustrated in Figure 2,
each domain requires the preparation and processing of data related to multiple associated capabilities. We carefully
construct and curate data for all capabilities associated with each domain. Multiple iterative experiments have validated
the effectiveness of our data construction strategy.
 
**Figure 2**

![](https://krseoul.imgtbl.com/i/2024/07/04/6686112fb6fb5.png)


### Long Context Extension
The long context capability of large models is very useful in many real-world application scenarios. After the initial
pre-training stage, we use an NTK-Aware incremental training method to modify the base value in the rotary position
encoding to 1,000,000, extending the context length that our model can handle from 4K to 32K. The incremental
training data is 40B tokens, with 67% of long text data longer than 16K and 33% of short text data shorter than 4K.

The long text fine-tuning data includes various tasks such as single document QA, multi-document QA, and document
summarization, which are mainly constructed through synthetic data. Taking the multi-document QA task as an example,
we select a short document data, extract the Q&A pair from it, and then randomly splice it with other short documents
to form a long text input longer than 8K. At the same time, we randomly require the answer to restate the corresponding
document number in the question to improve the model’s retrieval capability in long text tasks. Figure 3 shows the
results of the model on the “Needle in a Haystack” test.

**Figure 3**

![](https://krseoul.imgtbl.com/i/2024/07/04/6686112ea8a24.jpg)



## Training Strategy
### Data Mixing
In the SFT stage, we adopt a data mixing method based on weight proportion, configuring weights
according to factors such as the domain, difficulty level, and contribution to model capability, ensuring full coverage of
various capability points during the training process.

### Training Efficiency 
Due to the varying lengths of data in the SFT stage, there will be a large number of padded
tokens in the batch during training, wasting computing resources. To solve this problem, we use the packing strategy
to reorganize the data and ensure that there would be no cross-contamination between samples after packing through
attention masking and position encoding reset techniques. Our experiments show that the training efficiency was
significantly improved using this strategy, with the training time reduced to about 1/4 of the original time.

### Hyper-parameter Settings 
In the SFT stage, we adopt AdamW as the optimizer with β1 = 0.9, β2 = 0.98, and
epsilon =1e-8. We implement a cosine learning rate decay strategy with a warm-up period of 100 steps and a maximum
learning rate of 5e-6. The training rounds are set as 3, and only the loss of the response is calculated during the process.
In the DPO stage, we set the beta value to 0.1 and adjusted the learning rate to 5e-7, with the rest of the parameters
remaining unchanged.



## Evaluation

In this section, we conduct a comprehensive evaluation and analysis of the performance of the chat model JIUTIAN-139MoE-Chat across diverse domains, encompassing both widely adopted public
benchmarks and self-built benchmarks tailored for specific industrial applications and safety considerations. Specifically,
the public benchmarks involve tasks covering language and knowledge understanding, reasoning, mathematics, and
coding. Meanwhile, the self-built benchmarks originate from several critical industrial applications, such as industry specific courses, skill-level examinations, and job interviews. All evaluations on the open benchmarks are conducted
using OpenCompass [Contributors, 2023], a fair, open-source, and comprehensive platform designed for evaluating
large models, ensuring a standardized and consistent assessment across various models.

### Open Benchmarks
We evaluate our models on public benchmarks covering multiple domains, including language and knowledge understanding, reasoning, mathematics, and coding, which are commonly used to assess the capability of large language
model. More details about task description and experimental setup are listed as follows:


**General Knowledge Understanding**

We conduct evaluations based on a series of comprehensive examinations
to assess the capability of our models to understand language and general knowledge. There are 5 classical bench-marks for evaluation: MMLU [Hendrycks et al., 2020], C-Eval [Huang et al., 2024], CMMLU [Li et al., 2023],
GaokaoBench [Zhang et al., 2023], AgiEval [Zhong et al., 2023]. We report 5-shot results for MMLU, C-Eval, and
CMMLU. While for GaokaoBench and AgiEval, we report 0-shot results.

**Reasoning**

We adopt the challenging benchmark BBH [Suzgun et al., 2022] to access the reasoning ability of our
models. It contains 23 challenging tasks from BIG-Bench, where contemporary language models had not surpassed
human performance at the time. We report the results based on a 3-shot approach.

**Mathematics**

Mathematical proficiency is an integral part of a model’s cognitive and computational ability. We
exploit GSM8K [Cobbe et al., 2021] and Math [Hendrycks et al., 2021] for evaluation, and the results for these
benchmarks are reported based on 4-shot and 0-shot settings separately.

**Coding**

We report the Pass@1 scores on HumanEval [Chen et al., 2021] based on 0-shot approach and the results
on MBPP [Austin et al., 2021] based on 3-shot approach to access the model’s coding proficiency.


• Subjective Knowledge: MMLU (5-shot) (Hendrycks et al., 2020 ), Ceval (5-shot)(Huang et al., 2023 ), Cmmlu (5-shot)(Li et al., 2023c ), GaokaoBench (0-shot)(Zhang et al., 2023b ), AgiEval (0-shot)(Zhong et al., 2023a )

• Reasoning: BBH (3-shot) (Suzgun et al.,2022)

• Math: GSM8K (4-shot)(Cobbe et al., 2021), Math (0-shot)(Hendrycks et al., 2021 )

• Code: Pass@1 scores on HumanEval (0-shot)(Chen et al., 2021), MBPP (3-shot)(Austin et al., 2021 )

We conduct a comprehensive evaluation of our JIUTIAN-139MoE-Chat model on a wide range of open benchmarks.
We also show our evaluation results on 3 open-source chat models: Qwen-14B-Chat[Qwen-Team, 2023], Baichuan2-
13B-Chat[Yang et al., 2023], LLaMA2-13B-Chat[Touvron et al., 2023a], and the proprietary model GPT3.5[OpenAI,
2023].

It turns out that JIUTIAN-139MoE-Chat performs better than other models on most benchmarks (see Table 3 & Table 4). As we can see, GPT3.5
achieves the best performance on both GSM8k and MBPP. However, JIUTIAN-139MoE-Chat performs better than all
three open-source language models on these two benchmarks. As for the GaokaoBench benchmark, Qwen-14B-Chat
performs the best. We can see that JIUTIAN-139MoE-Chat still obtains a relatively high score compared to other
models.


**Table 3**


| Models| Mmlu |Ceval|Cmmlu |AgiEval |GaokaoBench|
| ------- | ------- |------- |------- |------- |------- |
|  Qwen-14B-Chat       |  66.4      |   71.7      |   70.0      |    47.3     |  **76.5**       |
|  Baichuan2-13B-Chat      | 50.5      |   53.4     |   50.7   |   32.2    | 40.9    |
|  LLaMA2-13B-Chat       |  	54.6       |   36.2      |   38.7      |    32.3   |  18.6     |
|  GPT3.5       |  	69.1       |  52.5      |   53.9     |    39.9   |  51.1   |
|  JIUTIAN-139MoE-Chat      |  	**82.5**      |  **88.1**      |  **87.4**    |    **77.9**   | 61.6  |



**Table 4**

| Models | GSM8k | Math | BBH | HumanEval |MBPP |
| ------- | ------- |------- |------- |------- |------- |
|  Qwen-14B-Chat       |   61.0      | 26.8       |   58.0     |   36.6   |  23.8    |    
|   Baichuan2-13B-Chat      |   36.3      |  7.3	      |       45.4      |  21.3      |     26.8   | 
|  LLaMA2-13B-Chat         | 37.1        |   5.2      |  40.2      |  18.9      |     27.2   | 
| GPT3.5       |  **78.2**      |28.0|    70.1    |   73.2     |60.2    |  
| JIUTIAN-139MoE-Chat  |   77.6   |  **53.2**      |   **91.0**  |      **70.1**    |**66.0**|



### Self-built Industry-Standard Dataset
An essential characteristic of our JIUTIAN-139MoE-Chat model, is its outstanding performance on industrial domain specific tasks. Specifically, to enhance our model’s industry-specific capabilities, we train JIUTIAN-139MoE-Chat with
tremendous data obtained from diverse industrial domains, including communication, electric power, transportation,
energy, steel, construction, etc.

We conduct evaluations with a self-built test dataset comprising industrial data closely related to human livelihood
and welfare. Sources of the test data include industry-specific courses, skill-level examinations, and job interview
10
JIUTIAN-139MoE TECHNICAL REPORT
questions. After rigorous cleaning and screening, we randomly select 200 objective multiple-choice questions for
accuracy evaluation in each industry, ensuring a balanced representation across industry subdivisions and difficulty
levels. Below is the list of industries against which we evaluate our model:

• Electric Power: Data on various aspects of the electrical industry, spanning from electrical safety, maintenance
and monitoring, to dispatching and control. It also delves into the intricacies of power generation engineering,
transmission engineering, distribution engineering, as well as analog and digital electronics technology.
Furthermore, it incorporates circuitry and electromagnetism principles, fundamental electrical knowledge, and
the skills required for electrical appliance maintenance.

• Steel Industry: Data on steel smelting, steel casting, rolling techniques, alloy materials, steel inspection,
welding processes, steel production safety, and the evolution of the steel industry.

• Aerospace: Data on aerospace engineering, including aerospace structures, avionics, aerodynamics, and
aviation meteorology. It delves into the intricacies of aircraft design and manufacturing, as well as aviation
transportation management. Additionally, it covers crucial aspects such as spaceflight safety management,
aviation maintenance management, and aviation materials.

• Construction: Data on architectural history and a range of professional roles, including registered safety
engineers, registered constructors, registered cost engineers, registered supervision engineers, safety officers,
material controllers, machinery operators, construction technicians, testing personnel, quality controllers, fire
protection engineers, environmental engineers, consulting engineers, BIM modelers.

• Finance: Data on a diverse range of professionals including accountants, auditors, tax advisors, economists,
statisticians, asset evaluators, risk managers, market analysts, international trade specialists, securities investors,
monetary bankers, and those with a foundation in economic theory.

• Energy: Data on clean coal technology, nuclear power, solar energy, biomass energy, hydropower, wind
energy, geothermal energy, hydrogen energy, petrochemical engineering and other emerging energy sources.

• Judiciary: Data on a broad range of disciplines, including constitutional law, criminal law, civil law, commercial law, intellectual property law, economic law, labor law, environmental and resource law, administrative
law, procedural law, international law, legal history, as well as legal ethics and professional responsibilities.

• Telecommunications: Data on fundamental communication technologies, wired communication techniques,
wireless communication systems, network communication protocols, data transmission methodologies, com-munication standards and protocols, the design and implementation of communication systems, communication
security and encryption techniques, communication software and applications, communication hardware and
equipment, optical communication technologies, as well as emerging communication technologies.

• Firefighting: Data on fire engineers, operators of fire-fighting facilities, firefighters, the science of fire
combustion, fire safety regulations, fire prevention and explosion protection techniques, fire safety management,
and the execution of fire engineering projects.

• Medical Industry: Data on a wide range of medical specialties, including internal medicine, surgery, obstetrics
and gynecology, pediatrics, otorhinolaryngology (head and neck surgery), dentistry, ophthalmology, traditional
Chinese medicine, dermatology, pathology, ultrasonography, laboratory medicine, rehabilitation medicine,
clinical nutrition, and medical psychology.
By evaluating our model on this diverse set of industry-standard datasets, we aim to ensure that our JIUTIAN-139MoE-Chat model possesses fundamental capabilities across various industries, positioning it as a powerful tool for various industry-specific applications.

**Table 5**

| Models            | Electric Power         | Steel Industry    | Aerospace    | Construction        | Finance       |
|-----------------|--------------|--------------|--------------|--------------|--------------|
| Qwen-14B-Chat   | 84.0           | 70.5           | 83.0           | 53.5           | 58.5           |
| Baichuan2-13B-Chat | 30.0          | 25.5           | 45.5           | 25.5           | 25.5           |
| LLaMA2-13B-Chat | 16.5           | 14.0           | 18.0           | 19.5           | 14.0           |
| GPT3.5      | 47.0           | 51.0           | 72.0           | 36.5           | 34.5           |
| JIUTIAN-139MoE-Chat | **91**     | **85**     | **94**     | **86**     | **81**     |
 
**Table 6**

|Models             | Energy       | Judiciary       | Telecommunications      | Firefighting       | Medical Industry     |
|-----------------|------------|------------|------------|------------|--------------|
| Qwen-14B-Chat   | 77.5         | 40.5         | 71.5        | 58.5         | 62.5           |
| Baichuan2-13B-Chat | 62.0       | 28.0         | 30.5         | 24.5         | 37.5           |
| LLaMA2-13B-Chat | 27.0         | 15.5         | 9.0          | 16.0         | 21.0           |
| GPT3.5      | 64.0         | 28.0         | 46.5         | 42.0         | 38.5           |
| JIUTIAN-139MoE-Chat | **88**     | **82**     | **81**     | **85**     | **91**     |






## Safety 


Following other state-of-the-art large models, we undertake a range of safety evaluations on our JIUTIAN-139MoE
model. In this section, we focus on evaluating the security capabilities of our model and other current powerful LLMs,
including content security and instruction security.

Content security mainly covers five major categories listed in the "Basic Requirements for Security of Generative
Artificial Intelligence Services [SAC/TC260]", including Chinese values, discrimination, commercial illegalities,
infringement of others’ rights, and reliability. We further refine and self-build upon these categories by creating 68
subcategories and 35000 pieces of evaluation data in Chinese.

Instruction security encompasses 30 attack methods across 15 types of vulnerabilities, including role-playing, reverse
engineering, privilege escalation, token manipulation, and other tactics.

We use a security score to calculate the percentage of security responses in the total number of responses, and the value
is positively correlated with the security of the model:

Score=Num(safe-response)÷Num（all-reponse）*100

**Table 7**

| Models | Chinese-values|Discrimination | Commercial_illegality| infringement |Reliability| Instruction- security |
| ------- | ------- |------- |------- |------- |------- |------- |
|  LLaMA2-13B-Chat       |    18.6     | 8.6        |   45.5      | 22.0        |     23.5    |     40.1    |  
| Baichuan2-13B-Chat   |42.8	 |45.2 |80.5|60.0 |83.5|59.2|
| Qwen-14B-Chat |70.8	 |76.1|**90.0**|**76.5**|**96.5**|77.9|
| JIUTIAN-139MoE-Chat |**80.2**|**84.2**|83.0|70.0|73.0|**80.8**|
									

We evaluate our model on our safety benchmarks, using the internal evaluation framework, and compare the results
with other open-source LLMs: LLaMA2-13B-Chat [Touvron et al., 2023a], Qwen-14B-Chat [Bai et al., 2023], and
Baichuan2-13B-Chat [Yang et al., 2023]. Table 7 shows the results.

The first five categories in Table 7 present the comparison results of content security. We can observe that JIUTIAN-
139MoE-Chat achieves consistently decent performance across all categories and is significantly superior to other
models of discrimination prevention and consistency with Chinese values. In terms of Infringement, it performs slightly
inferior to the best model Qwen-14B-Chat but significantly surpasses other models. Regarding reliability and avoiding
commercial illegalities, our JIUTIAN-139MoE-Chat also performs comparably to the first tier. For instruction security,
results from the last column of Table 7 demonstrate our absolute advantage among all the models involved in the
comparison. Overall, the JIUTIAN-139MoE-Chat performs consistently well when being evaluated from various
aspects of security.



## Inference

Please refer to [**tutorial**](./docs/tutorial.md) for the model's inference operations.


```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_id = "/models/JIUTIAN-139MoE-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
text = "Please introduce the Great Wall."
text = "Human:\n" + text + "\n\nAssistant:\n"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False,padding_side='left',truncation_side='left')
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.03,do_sample=False,eos_token_id=0)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))
```




## Statement

We hereby declare that the JIUTIAN model and its derivative models shall not be used for any activities that may endanger national and social security or violate the law. At the same time, we require users not to use the JIUTIAN  model for internet services that have not undergone safety review and filing. We hope that all users will abide by the above principles to ensure that technological development takes place in a legal and compliant environment.

We have done our best to ensure the compliance of the data used during the model training process. However, despite our great efforts, due to the complexity of the model and data, there may still be some unforeseeable issues. Therefore, we will not assume any responsibility for any issues arising from the use of the JIUTIAN open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, abused, disseminated, or misused.

## License
The use of the JIUTIAN model by the community must comply with the [JIUTIAN Large Model Community License Agreement](./JIUTIAN_Large_Model_Community_License_Agreement.pdf). The JIUTIAN model supports commercial use; if you plan to use the JIUTIAN model or its derivatives for commercial purposes, you are required to submit the application materials required by the [JIUTIAN Large Model Community License Agreement](./JIUTIAN_Large_Model_Community_License_Agreement.pdf) through the following contact email: jiutian_ai@chinamobile.com. After the review is approved, a non-exclusive, worldwide, non-transferable, non-sublicenseable, revocable commercial copyright license will be specifically granted to you.



<br>
