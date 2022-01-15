# Towards Fast and Transferable Black-box Adversarial Attacks

## Anonymous submission

<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/figure2.PNG" width="100%" height="100%">

## Installation
0. The CUDA and Pytorch version that is used for this work:
~~~
'CUDA==11.0',
'torch==1.7.1',
~~~

1. Installation
~~~
git clone https://github.com/anonymous202201/fast_transferable_blackbox_attack.git
cd fast_transferable_blackbox_attack
pip install -e .
~~~

2. Examples of Training and Evaluation

Train:
~~~
python tools/generate.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --dataset_type $DATASET_TYPE --step_size $STEP_SIZE --num_steps $NUM_STEPS --attack_method $ATTACK_METHOD --surrogate_model $SURROGATE_MODEL --custimized_pretrain $CUSTIMIZED_PRETRAIN
~~~
Eval:
~~~
python tools/evaluate.py
~~~

## Main Results

<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table1.PNG" width="100%" height="100%">
<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table2.PNG" width="100%" height="100%">
<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table3.PNG" width="50%" height="50%">
