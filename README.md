## AdvRPPRL: Towards Adversarially Robust and Privacy-Preserving Representation Learning via Information Theory
Our implementation is based on code from Learning Adversarially Robust Representations via Worst-Case Mutual Information
Maximization [(ICML 2020)](https://arxiv.org/abs/2002.11798) and our team's privacy presentation via information theory 
concept. For all the scripts, we assume the working directory to be the root folder of our code. You can reference the
read.me file on [(Git-hub)](https://github.com/schzhu/learning-adversarially-robust-representations) for more information.
#### Train a basic model

We adopt the training structure of the Sicheng Zhu, Xiao Zhang, and David Evans' robustness representation project,
so you still can train basic or robust encoders and classifiers for reference.

#### Encoder-classifier 2 step training

step-1, encoder
```bash
python main.py --dataset cifar \
--task train-encoder \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--representation-type layer \
--estimator-loss worst \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--exp-name learned_encoder
```

step-2, classifier
```bash
python main.py --dataset cifar \
--task train-classifier \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--classifier-arch mlp \
--representation-type layer \
--classifier-loss robust \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.latest \
--exp-name test_learned_encoder
```
&nbsp;

#### Our training path
Since we adopt the basic training structure, most of the training steps remain the same, but we have added a few arguments
* -\-estimator-loss = **worst** (worst-case mutual information estimation) / **normal** (normal-case mutual information estimation)

For example, to test the worst-case mutual information of ResNet18, run
```bash
python main.py --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--task estimate-mi \
--representation-type layer \
--estimator-loss worst \
--arch resnet18 \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.best \
--exp-name estimator_worst__resnet18_adv \
--no-store
```
or to test on the baseline-h, run
```bash
python main.py --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--task estimate-mi \
--representation-type layer \
--estimator-loss worst \
--arch baseline_mlp \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.best \
--exp-name estimator_worst__baseline_mlp_adv \
--no-store
```
&nbsp;

#### Learn Representations
Set **task=train-encoder** to learn a representation using our training principle. For train by worst-case mutual information maximization, we can use other lower-bound of mutual information as surrogate for our target, which may have slightly better empirical performance (e.g. nce). Please refer to arxiv.org/abs/1808.06670 for more information.
Optional commands:
* -\-estimator-loss = **worst** (worst-case mutual information maximization) / **normal** (normal-case mutual information maximization)
* -\-va-mode = **dv** (Donsker-Varadhan representation) / **nce** (Noise-Contrastive Estimation) / **fd** (fenchel dual representation)
* -\-arch = **basic_encoder** ([Hjelm et al.](https://arxiv.org/abs/1808.06670)) / ...


Example:
```bash
python main.py --dataset cifar \
--task train-encoder \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--representation-type layer \
--estimator-loss worst \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--exp-name learned_encoder
```
&nbsp;

#### Test on Downstream Classifications (Figure 4, 5, 6; Table 1, 3)
Set **task=train-classifier** to test the classification accuracy of learned representations. 
Optional commands:
* -\-classifier-loss = **robust** (adversarial classification) / **standard** (standard classification)
* -\-classifier-arch = **mlp** (mlp as downstream classifier) /  **linear** (linear classifier as downstream classifier)

Example:
```bash
python main.py --dataset cifar \
--task train-classifier \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--classifier-arch mlp \
--representation-type layer \
--classifier-loss robust \
--epoch 500 --lr 1e-4 --step-lr 10000 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/saved/model/checkpoint.pt.latest \
--exp-name test_learned_encoder
```

