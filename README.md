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
Since we adopt the basic training structure, most of the training steps remain the same, but we have added a few arguments.
Training happens in the order of, robust, privacy attack, privacy protection, utility, then all-loss, under each epoch.
* -\-task = **train-all** (suggest to use train-all, set parts that aren't needed to 0)
* -\-robust_rounds = **0** (# of sub-rounds of original training rounds by Sicheng Zhu, Xiao Zhang, and David Evans)
* -\-privacy_cla = **0** (# of sub-rounds of privacy cla inference attack training)
* -\-privacy_rounds = **0** (# of sub-rounds of privacy defense training)
* -\-utility_rounds = **0** (# of sub-rounds of JSD utility, similar to robust_rounds function, can elect to run 0)
* -\-all_rounds = **0** (# of sub-rounds of all-loss based training)
* -\-alpha = **0** (alpha set between 0 and 1 to determine the weight given to robust loss)
* -\-beta = **0** (alpha set between 0 and 1 to determine the weight given to privacy loss)

For example, to train encoder under the worst-case mutual information of ResNet18, run
```bash
python main.py --dataset cifar \
--data /path/to/dataset \
--out-dir /path/to/output \
--task train-all \
--representation-type layer \
--estimator-loss worst \
--arch resnet18 \
--epoch 20 --robust_rounds 0 --privacy_cla 0 --privacy_rounds 0 \
--utility_rounds 0 --all_rounds 1 --alpha 0.4 --beta 0.6 \
--lr 1e-3 --step-lr 100 --workers 2 \
--attack-lr 1e-2 --constraint inf --eps 8/255 \
--exp-name estimator_worst__resnet18_adv
```

#### Test on Downstream Classifications
Set **task=train-classifier** to test the classification accuracy of learned representations. 
Optional commands:
* -\-task = train-classifier (train cla) / train-privacy_classifier (inference attack, only mlp)
* -\-classifier-loss = **robust** (adversarial cla) / **standard** (standard cla)
* -\-classifier-arch = **mlp** (mlp as downstream classifier) /  **linear** (linear classifier as downstream classifier)

Example:
```bash
python main.py --dataset cifar \
--task train-privacy_classifier \
--data /path/to/dataset \
--out-dir /path/to/output \
--arch basic_encoder \
--classifier-arch mlp \
--representation-type layer \
--classifier-loss standard \
--epoch 50 --lr 1e-3 --step-lr 100 --workers 2 \
--attack-lr=1e-2 --constraint inf --eps 8/255 \
--resume /path/to/output/estimator_worst__resnet18_adv/checkpoint.pt.latest \
--exp-name test_learned_encoder
```

