### Code for CS598 Final Project

Group ID: 119
Paper ID: 40

### A. Citation to the original paper

G. Lee, H. Hwang, S. Bae, Y. Kwon, W. Shin,
S. Yang, M. Seo, J.-Y. Kim, and E. Choi. Ehrsql: A
practical text-to-sql benchmark for electronic health
records. Advances in Neural Information Processing
Systems, 35:15589–15601, 2022

https://openreview.net/forum?id=B2W8Vy0rarw

### B. Link to the original paper’s repo

https://github.com/glee4810/EHRSQL

### C. Reproducing paper results with baseline model in the paper using Google Colab notebook on GPU device:

1. **Dependencies**

```python
!pip install pandas
!pip install dask
!pip install wandb
!pip install nltk
!pip install scikit-learn
!pip install func-timeout
!pip install transformers
!pip install sentencepiece
```

2. **Mount Google drive**
   Mount Google Drive for permanent data keeping.
   Colab storage is temporary.

```python
# import drive from google colab
from google.colab import drive
# default location for the drive
ROOT = "/content/drive"
# mount the google drive at /content/drive
drive.mount(ROOT)
```

3. **Clone Repository**

```python
# change directory
%cd /home/
!git clone https://github.com/melindatsay/EHRSQL.git
%cd EHRSQL
```

4. **Data download instruction**
   Use terminal to download data through PhysioNet’s credential access

- **eICU**
  https://physionet.org/content/eicu-crd/2.0/

```python
wget -r -N -c -np --user {  } --ask-password https://physionet.org/files/eicu-crd/2.0/
```

- **MIMIC-III**
  https://physionet.org/content/mimiciii/1.4/

```python
wget -r -N -c -np --user {  } --ask-password https://physionet.org/files/mimiciii/1.4/
```

5. **Unzip data gz file**

```python
# unzip eicu dataset
%cd /content/physionet.org/files/eicu-crd/2.0/
!gzip -d *.gz
#unzip mimic-iii dataset
%cd /content/physionet.org/files/mimiciii/1.4/
!gzip -d *.gz
```

6. **Preprocessing data**

```python
%cd /home/EHRSQL/preprocess
# preprocessing eicu dataset
!python3 preprocess_db.py --data_dir /content/physionet.org/files/eicu-crd/2.0/ --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
# preprocessing mimic-iii dataset
!python3 preprocess_db.py --data_dir /content/physionet.org/files/mimiciii/1.4/ --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
```

7. **Training T5-base model**

```python
# find gpu_uuid
!nvidia-smi -q
# train model using eicu dataset
!python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_eicu_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES {gpu_uuid}
# train model using mimic-iii dataset
!python T5/main.py --config T5/config/ehrsql/training/t5_ehrsql_mimic3_natural_lr0.001.yaml --CUDA_VISIBLE_DEVICES {gpu_uuid}
```

Once the training process is done, the pre-trained model is saved in /home/EHRSQL/outputs/

8. **Pre-trained model**

- Download pre-trained model using eicu dataset from the following link.
  Add the folder, **t5_ehrsql_eicu_natural_lr0.001**, to **/EHRSQL/outputs/**
  (Noted that only **checkpoint_best.pth.tar** is used for predicting SQL queries.)
  https://drive.google.com/drive/folders/1qyi9YhVpc3V6emiSPkzJaMBQDcMPr9WQ?usp=sharing
  <br>
- Download pre-trained model using mimic-iii dataset from the following link.
  Add the folder, **t5_ehrsql_mimic3_natural_lr0.001**, to **/EHRSQL/outputs/**
  (Noted that only **checkpoint_best.pth.tar** is used for predicting SQL queries.)
  https://drive.google.com/drive/folders/1SJAib_pLResdbIasi2FkYvZheWKRvb_8?usp=sharing

9. **Generate SQL queries**

```python
# find gpu_uuid
!nvidia-smi -q
# for eicu dataset
!python T5/main.py --config T5/config/ehrsql/eval/ehrsql_eicu_t5_base__eicu_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_uuid>
!python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_ehrsql_eicu_t5_base__eicu_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
# for mimic-iii dataset
!python T5/main.py --config T5/config/ehrsql/eval/ehrsql_mimic3_t5_base__mimic3_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_uuid>
!python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
```

10. **Evaluation**

```python
# for eicu
!python evaluate.py --db_path ./dataset/ehrsql/eicu/eicu.db --data_file dataset/ehrsql/eicu/valid.json --pred_file ./outputs/eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid/prediction.json
# for mimic-iii
!python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid/prediction.json
```

11. **Table of Results**

- **eICU dataset, valid sets**
  | Result | F1ans| Pexe | Rexe |F1exe|
  |--------|------|------|------|-----|
  | Paper | 93.0 | 92.5 | 91.7 | 92.1|
  | Reproduced | 92.4 | 93.1 | 89.1 | 91.1|
- **MIMIC-III dataset, valid sets**
  | Result | F1ans| Pexe | Rexe |F1exe|
  |--------|------|------|------|-----|
  | Paper | 94.8 | 94.1 | 93.2 | 93.7|
  | Reproduced | 94.3 | 91.6 | 91.5 | 91.5|

<br>

### D. Ablation Study: Reduce training time by refactoring the training process and utilizing Google Cloud TPUs (Tensor Processing Units) to achieve multi-device and multi-process training

Noted that the notebook is over 100 MB, which cannot be uploaded on Github.

The following 1-10 steps and implementations of training process can be found in the following Google Colab Notebook.
https://colab.research.google.com/drive/1V0-sBYxLKv6cRdLidiIuH_hH86nPipx6?usp=sharing

<br>

1.  **Apply FREE TPU Access From Google**

    - Apply at https://sites.research.google/trc/about/
    - It will take several hours to receive feedback from Google

2.  **Requirements**

    - Python version == 3.9
    - Pytorch version == 2.0
    - Torch_xla version:
      https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp39-cp39-linux_x86_64.whl
    - TPU version == TPU V2-8

3.  **Select TPU setting in Google Colab Notebook**

        - Open "change runtime type" from "Runtime"
        - Select "TPU" and "High-Ram" settings

4.  **Install Colab compatible PyTorch/XLA wheels in Google Colab Notebook**
    Remember to use the fallback runtime version on Colab for compatibility:

        - Open "Command Palette" from "Tools" or Ctrl+Shift+P
        - Select "Use fallback runtime version"
        - Wait for runtime to re-initiate

5.  **Dependencies**

```python
!pip install func-timeout
!pip install transformers
!pip install sentencepiece
!pip install openai
!pip install cloud-tpu-client==0.10 cloud-tpu-profiler torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp39-cp39-linux_x86_64.whl
!pip install -U tensorboard-plugin-profile
```

6. **Mount Google drive**
   Mount Google Drive for permanent data keeping.
   Colab storage is temporary.

```python
# import drive from google colab
from google.colab import drive
# default location for the drive
ROOT = "/content/drive"
# mount the google drive at /content/drive
drive.mount(ROOT)
```

7. **Clone Repository**

```python
# change directory
%cd /home/
!git clone https://github.com/melindatsay/EHRSQL.git
%cd EHRSQL
```

8. **Data download instruction**
   Use terminal to download data through PhysioNet’s credential access

- **eICU**
  https://physionet.org/content/eicu-crd/2.0/

```python
wget -r -N -c -np --user {  } --ask-password https://physionet.org/files/eicu-crd/2.0/
```

- **MIMIC-III**
  https://physionet.org/content/mimiciii/1.4/

```python
wget -r -N -c -np --user {  } --ask-password https://physionet.org/files/mimiciii/1.4/
```

9. **Unzip data gz file**

```python
# unzip eicu dataset
%cd /content/physionet.org/files/eicu-crd/2.0/
!gzip -d *.gz
#unzip mimic-iii dataset
%cd /content/physionet.org/files/mimiciii/1.4/
!gzip -d *.gz
```

10. **Preprocessing data**

```python
%cd /home/EHRSQL/preprocess
# preprocessing eicu dataset
!python3 preprocess_db.py --data_dir /content/physionet.org/files/eicu-crd/2.0/ --db_name eicu --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1
# preprocessing mimic-iii dataset
!python3 preprocess_db.py --data_dir /content/physionet.org/files/mimiciii/1.4/ --db_name mimic_iii --deid --timeshift --current_time "2105-12-31 23:59:00" --start_year 2100 --time_span 5 --cur_patient_ratio 0.1 &
```

11. **Training T5-base model on Google Cloud TPU**

- The 1-10 steps and implementations of training process can be found in the following Google Colab Notebook. (Noted that the notebook is over 100 MB, which cannot be uploaded on Github.)
  https://colab.research.google.com/drive/1V0-sBYxLKv6cRdLidiIuH_hH86nPipx6?usp=sharing
- The default dataset used for training model is eICU. To use the MIMIC-III dataset for training model, change the config_file to "T5/config/ehrsql/training/t5_ehrsql_mimic3_natural_lr0.001.yaml"
- The traing logs are recorded in /home/EGRSQL/training_logs_stdout
- Once the training process is done, the pre-trained model is saved in /home/EHRSQL/outputs/

12. **Pre-trained model**

- Download pre-trained model using eicu dataset from the following link.
  Add the folder, **t5_ehrsql_eicu_natural_lr0.001**, to **/EHRSQL/outputs/**
  (Noted that only **checkpoint_best.pth.tar** is used for predicting SQL queries.)
  https://drive.google.com/drive/folders/1qyi9YhVpc3V6emiSPkzJaMBQDcMPr9WQ?usp=sharing
  <br>
- Download pre-trained model using mimic-iii dataset from the following link.
  Add the folder, **t5_ehrsql_mimic3_natural_lr0.001**, to **/EHRSQL/outputs/**
  (Noted that only **checkpoint_best.pth.tar** is used for predicting SQL queries.)
  https://drive.google.com/drive/folders/1SJAib_pLResdbIasi2FkYvZheWKRvb_8?usp=sharing

13. **Generate SQL queries**

```python
# find gpu_uuid
!nvidia-smi -q
# for eicu dataset
!python T5/main.py --config T5/config/ehrsql/eval/ehrsql_eicu_t5_base__eicu_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_uuid>
!python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_ehrsql_eicu_t5_base__eicu_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
# for mimic-iii dataset
!python T5/main.py --config T5/config/ehrsql/eval/ehrsql_mimic3_t5_base__mimic3_valid.yaml --output_file prediction_raw.json --CUDA_VISIBLE_DEVICES <gpu_uuid>
!python T5/abstain_with_entropy.py --infernece_result_path outputs/eval_ehrsql_mimic3_t5_base__mimic3_valid --input_file prediction_raw.json --output_file prediction.json --threshold 0.14923561
```

14. **Evaluation**

```python
# for eicu
!python evaluate.py --db_path ./dataset/ehrsql/eicu/eicu.db --data_file dataset/ehrsql/eicu/valid.json --pred_file ./outputs/eval_t5_ehrsql_eicu_natural_lr0.001_best__eicu_natural_valid/prediction.json
# for mimic-iii
!python evaluate.py --db_path ./dataset/ehrsql/mimic_iii/mimic_iii.db --data_file dataset/ehrsql/mimic_iii/valid.json --pred_file ./outputs/eval_t5_ehrsql_mimic3_natural_lr0.001_best__mimic3_natural_valid/prediction.json
```

15. **Table of Results**
    The results are the same as the one in section C.11, which is expected since the implementation incurred in ablation study contributes to the training process speed instead of modification of model itself. <br>
    The training process has been refactored for utilizing Google Cloud TPUs to achieve multi-device and multi-process training. The result of training time on TPU is listed below. Unexpectedly, it takes similar time to train T5-base model on TPUs compared to GPU under standard RAM settings. The configuration still requires some fine-tuning to accommodate with TPUs to release the computing power to reduce training time.

| Datasets Used | TPU model         | TPU hours |
| ------------- | ----------------- | --------- |
| eICU          | Google Cloud TPUs | 14.35 hrs |
| MIMIC-III     | Google Cloud TPUs | 12 hrs    |
