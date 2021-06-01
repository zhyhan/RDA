# Robust Domain Adaptation 

Code release for "Towards Accurate and Robust Domain Adaptation under Noisy Environments"
## Prerequisites:

* Python == 3.7
* PyTorch ==1.8.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.9.01
* Numpy
* argparse
* easydict
* pillow = 2.3.5
* tqdm

## Dataset:

You need to modify the path of the image in every ".txt" in "./data".
The COVID-19 dataset can be downloaded at the [BaiduCloud](https://pan.baidu.com/s/17KEGkPmue6jBRRq6pGVF4Q) and the code is c8kk.

## Training:

You can run "./scripts/train.sh" to train and evaluate on the task. Before that, you need to change the project root, dataset (Office-home or Office-31), data address and CUDA_VISIBLE_DEVICES in the script.

## Citation:

If you use this code for your research, please consider citing:
```
@inproceedings{ijcai2020-314,
  title     = {Towards Accurate and Robust Domain Adaptation under Noisy Environments},
  author    = {Han, Zhongyi and Gui, Xian-Jin and Cui, Chaoran and Yin, Yilong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {2269--2276},
  year      = {2020},
  month     = {7},
  note      = {Main track},
  doi       = {10.24963/ijcai.2020/314},
  url       = {https://doi.org/10.24963/ijcai.2020/314},
}
```
## Contact
If you have any problem about our code, feel free to contact hanzhongyicn@gmail.com
