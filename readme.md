# HAROOD: A Benchmark for Out-of-distribution Generalization in Sensor-based Human Activity Recognition

HAROOD is a modular and reproducible benchmark framework for studying generalization in sensor-based human activity recognition (HAR).  
It unifies preprocessing pipelines, standardizes four realistic OOD scenarios, and implements 16 representative algorithms across CNN and Transformer architectures.

This repository contains the official implementation accompanying the paper  
"[HAROOD: A Benchmark for Out-of-distribution Generalization in Sensor-based Human Activity Recognition](https://arxiv.org/abs/2512.10807)" published in **KDD 2026**.

---

## ⭐ Key Features

- 6 public HAR datasets
- 5 realistic OOD scenarios (cross-person, cross-position, cross-dataset, cross-time, cross-device)
- 16 generalization algorithms
- CNN and Transformer backbones
- Standardized train/val/test splits
- Easy extensibility (new datasets, new algorithms)

---

# 1. Installation

We recommend using **conda** to build a reproducible environment.

## 1.1 Create the environment

```bash
conda env create -f environment.yml
conda activate myenv
```

# 2. Project Structure

```text
harood/
│── alg/           # 16 OOD methods
│── network/       # CNN & Transformer models
│── datautil/      # Dataset loaders
│── analys/        # Performance analysis
```

---

# 3. Usage

## 3.1 Download datasets

Download the dataset from [here](https://huggingface.co/datasets/AIFrontierLab/HAROODdata) and extract it into the current directory.

## 3.2 Run with a YAML config

```python
from core import train
results = train(config='./config/experiment.yaml')
```

## 3.3 Run with a Python dict

```python
from core import train
config_dict = {
    'algorithm': 'CORAL',
    'batch_size': 32,
}
results = train(config=config_dict)
```

## 3.4 Override parameters

```python
results = train(
    config='./config/experiment.yaml',
    lr=2e-3,
    max_epoch=200,
)
```

---


# 4. Supported Algorithms

We currently support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

### Data Manipulation
- Mixup [2]
- DDLearn [3]

### Representation Learning
- ERM [1]
- DANN [4]  
- CORAL [5]
- MMD [6]
- VREx [7]
- LAG [8]


### Learning Strategy
- MLDG [9]
- RSC [10]
- GroupDRO [11]  
- ANDMask [12]
- Fish [13]
- Fishr [14]
- URM [15]
- ERM++ [16]

---


# 5. Extending the Benchmark

## Add a new algorithm:

1. Add a file in `alg/algs`
2. Implement a class with `__init__`, `update`, `predict`
3. Register the algorithm

## Add a new dataset:

To add a new dataset, follow the standardized preprocessing pipeline:

1. **Segment raw sensor data into fixed-length windows**  
   Apply the unified sliding-window procedure (window size and step size used in HAROOD).

2. **Generate `x.npy` and `y.npy`**  
   - `x.npy`: windowed sensor data with shape *(N, T, C)*  
   - `y.npy`: labels for each window, including  
     - activity class  
     - position label  
     - sensor type label  

3. **Save the files**  
   The dataset folder should contain:
```
dataset_name/
    x.npy
    y.npy
```

4. **Register the dataset**  
Add the dataset entry in `util.py` so that HAROOD can automatically load it during training.

Once registered, the dataset can be used directly in any experiment configuration.

---

# 6. Contribution

The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first.

---

# 7. Acknowledgment

Great thanks to [DomainBed](https://github.com/facebookresearch/DomainBed) and [DeepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG). We simplify their work to make it easy to perform experiments and extensions. Moreover, we add some new features.

# 8. Reference

[1] Vladimir Vapnik and Vlamimir Vapnik. 1998. Statistical learning theory Wiley. New York 1, 624 (1998), 2.

[2] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2018. mixup: Beyond Empirical Risk Minimization. In International Conference on Learning Representations.

[3] Xin Qin, Jindong Wang, Shuo Ma, Wang Lu, Yongchun Zhu, Xing Xie, and Yiqiang Chen. 2023. Generalizable low-resource activity recognition with diverse and discriminative representation learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1943–1953.

[4] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. Journal of machine learning research 17, 59 (2016), 1–35.

[5] Baochen Sun and Kate Saenko. 2016. Deep coral: Correlation alignment for deep domain adaptation. In European conference on computer vision. Springer, 443–450.

[6] Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. 2018. Domain generalization with adversarial feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition. 5400–5409.

[7] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. 2021. Out-of-distribution generalization via risk extrapolation (rex). In International conference on machine learning. PMLR, 5815–5826.

[8] Wang Lu, Jindong Wang, and Yiqiang Chen. 2022. Local and global alignments for generalizable sensor-based human activity recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 3833–3837.

[9] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy Hospedales. 2018. Learning to generalize: Meta-learning for domain generalization. In Proceedings of the AAAI conference on artificial intelligence, Vol. 32.

[10] Zeyi Huang, Haohan Wang, Eric P Xing, and Dong Huang. 2020. Self-challenging improves cross-domain generalization. In European conference on computer vision. Springer, 124–140.

[11] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. 2020. Distributionally Robust Neural Networks. In International Conference on Learning Representations.

[12] Giambattista Parascandolo, Alexander Neitz, Antonio Orvieto, Luigi Gresele, and Bernhard Schölkopf. 2021. Learning explanations that are hard to vary. In The 9th International Conference on Learning Representations (ICLR 2021). OpenReview.
net.

[13] Yuge Shi, Jeffrey Seely, Philip Torr, Awni Hannun, Nicolas Usunier, Gabriel Synnaeve, et al. 2022. Gradient Matching for Domain Generalization. In International Conference on Learning Representations

[14] Alexandre Rame, Corentin Dancette, and Matthieu Cord. 2022. Fishr: Invariant gradient variances for out-of-distribution generalization. In International Conference on Machine Learning. PMLR, 18347–18377.

[15] Kiran Krishnamachari, See-Kiong Ng, and Chuan-Sheng Foo. 2024. Uniformly distributed feature representations for fair and robust learning. Transactions on Machine Learning Research (2024).

[16] Piotr Teterwak, Kuniaki Saito, Theodoros Tsiligkaridis, Kate Saenko, and Bryan A Plummer. 2025. Erm++: An improved baseline for domain generalization. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 8525–8535.
 
---

# 9. Citation

If you use HAROOD, please cite:

```
@inproceedings{lu2026harood,
  title={HAROOD: A Benchmark for Out-of-distribution Generalization in Sensor-based Human Activity Recognition},
  author={Lu, Wang and Zhu, Yao and Wang, Jindong},
  booktitle={The 32rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2026}
}
```

