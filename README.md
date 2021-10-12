# AASIST

This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in ['AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'](https://arxiv.org/abs/2110.01200)

### Getting started
`requirements.txt` must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible. 
- Installing dependencies
```
pip install -r requirements.txt
```
- Our environment (for GPU training)
  - Based on a docker image: `pytorch:1.6.0-cuda10.1-cudnn7-runtime`
  - GPU: 1 NVIDIA Tesla V100
    - About 16GB is required to train AASIST using a batch size of 24
  - gpu-driver: 418.67

### Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset.
```
python ./download_dataset.py
```

### Training 
The `main.py` includes train/validation/evaluation.

To train AASIST [1]:
```
python main.py --config ./config/AASIST.conf
```
To train AASIST-L [1]:
```
python main.py --config ./config/AASIST-L.conf
```

#### Training baselines

We additionally enabled the training of RawNet2[2] and RawGAT-ST[3]. 

To Train RawNet2 [2]:
```
python main.py --config ./config/RawNet2_baseline.conf
```

To train RawGAT-ST [3]:
```
python main.py --config ./config/RawGATST_baseline.conf
```

### Pre-trained models
We provide pre-trained AASIST and AASIST-L.

To evaluate AASIST [1]:
- It shows `EER: 0.83%`, `min t-DCF: 0.0275`
```
python main.py --eval --config ./config/AASIST.conf
```
To evaluate AASIST-L [1]:
- It shows `EER: 0.99%`, `min t-DCF: 0.0309`
- Model has `85306` parameters
```
python main.py --eval --config ./config/AASIST-L.conf
```


### Developing custom models
Simply by adding a configuration file and a model architecture, one can train and evaluate their models.

To train a custom model:
```
1. Define your model
  - The model should be a class named "Model"
2. Make a configuration by modifying "model_config"
  - architecture: filename of your model.
  - hyper-parameters to be tuned can be also passed using variables in "model_config"
3. run python main.py --config {CUSTOM_CONFIG_NAME}
```

### License
```
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)

The repository for baseline RawGAT-ST model will be open
-  https://github.com/eurecom-asp/RawGAT-ST-antispoofing

### References
[1] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021},
  pages
```

[2] End-to-End anti-spoofing with RawNet2
```bibtex
@INPROCEEDINGS{Tak2021End,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}
```

[3] End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection
```bibtex
@inproceedings{tak21_asvspoof,
  author={Tak, Hemlata and Jung, Jee-weon and Patino, Jose and Kamble, Madhu and Todisco, Massimiliano and Evans, Nicholas},
  title={{End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection}},
  year=2021,
  booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  pages={1--8},
  doi={10.21437/ASVSPOOF.2021-1}
```
