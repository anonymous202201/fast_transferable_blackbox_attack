# Towards Fast and Transferable Black-box Adversarial Attacks

## Anonymous submission

## Abstract

Recent research on adversarial attacks has brought the vulnerabilities of neural networks into the spotlight. To make adversarial examples (AEs) more practical, researchers treat victim models as blackboxes, generate AEs on surrogate models and then transfer the effectiveness from the surrogates to victim models. However, these attack methods, although being transferable, are prone to having unsatisfactory attack success rates (ASR) and being time consuming. To address these issues, we propose the Fast Transferable Attack (FTA), which is an effective and efficient black-box and transferable adversarial attack. FTA employs Lipschitz constant-based loss function on an intermediate feature map and generates AEs to approach the Lipschitz constant point of the surrogate model. Instead of accessing all layers of a surrogate, FTA only needs to access from the input layer to an intermediate layer, which significantly accelerates the attack. Our research also shows that FTA can achieve higher ASR when the feature maps of the surrogates are sparser. Therefore, we propose Reg-ADMM, which incorporates the Alternating Direction Method of Multipliers (ADMM) into the Lipschitz-based loss function and sparsifies both the intermediate feature map and parameters. The sparsified surrogates decrease the attack latency and meanwhile increase the ASR of the proposed FTA. Compared to the state-of-the-art blackbox transferable attacks, proposed FTA achieves higher ASR and faster processing speed. Evaluations performed on the ImageNet and BDD100k datasets show that the FTA outperforms the stateof-the-art methodologies by a large margin.

<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/figure1.PNG" width="50%" height="50%">
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

### 3D Object detection on nuScenes

<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table1.PNG" width="100%" height="100%">
<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table2.PNG" width="100%" height="100%">
<img src="https://github.com/anonymous202201/fast_transferable_blackbox_attack/blob/main/docs/table3.PNG" width="50%" height="50%">
