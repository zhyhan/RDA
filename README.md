# Robust Domain Adaptation 

Code release for "Towards Accurate and Robust Domain Adaptation under Noisy Environments"
## Prerequisites:

* Python3
* PyTorch ==0.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.2.0
* Numpy
* argparse
* PIL
* tqdm

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".

## Training:

You can run "./scripts/train.sh" to train and evaluate on the task. Before that, you need to change the project root, dataset (Office-home or Office-31), data address and CUDA_VISIBLE_DEVICES in the script.

## Citation:

If you use this code for your research, please consider citing:
```
@article{han2020towards,
  title={Towards Accurate and Robust Domain Adaptation under Noisy Environments},
  author={Han, Zhongyi and Gui, Xian-Jin and Cui, Chaoran and Yin, Yilong},
  journal={arXiv preprint arXiv:2004.12529},
  year={2020}
}
```
## Contact
If you have any problem about our code, feel free to contact hanzhongyicn@gmail.com
