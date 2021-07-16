# Quantization-DNNs

simulate [TensorFlow Lite](https://arxiv.org/abs/1712.05877) with PyTorch API  
reference: <https://www.zhihu.com/people/xu-yong-zhe-65/posts>  
Since I just use Pytorch API to implement INT8 quantization, I cannot obtain the speed-up effect proposed in the paper, but just simulate the influence brought be the quantization operation.
- - -

## Usage

- train full-precision model

  ```python
  python ./train.py
  ```

- post training quantization

  ```python
  python ./PTQ.py
  ```

- quantization aware training

  ```python
  python ./QAT.py
  ```

## Experiment Results

- full-precision model  

   **FashionMNIST dataset**  

   |model type|accuracy|
  | :--: | :--: |
  |CNNs|90%|
  |CNNs_BN|90% |

    **CIFAR10 dataset**  

   |model type|accuracy|
  | :--: | :--: |
  | CNNs | 69% |
  | CNNs_BN | 73% |

- quant model

  **FashionMNIST dataset**  

  |num_bits | 4 | 6 | 8 |
  | :--: | :--: | :--: | :--: |
  | cnn_PTQ | 71% | 88% | 90% |
  | cnn_QAT | 81% | 90% | 91% |
  | cnnBN_PTQ | 52% | 63% | 59% |
  | cnnBN_QAT | 62% | 71% |  84%  |

  **CIFAR10 dataset**  

  |num_bits | 4 | 6 | 8 |
  | :--: | :--: | :--: | :--: |
  | cnn_PTQ | 47% | 67% | 69% |
  | cnn_QAT | 47% | 67% | 69%  |
  | cnnBN_PTQ | 42% | 45% | 57% |
  | cnnBN_QAT | 43% | 53% |  59%  |

## Observation

- **The quantization precision of bias vectors has huge influence to the final results**
  > As proposed in the paper, the use of higher precision(int32) of bias vectors meets the real need since they are added to many activations. The following experiments verify the explanation:  

- **Learning rate sometimes may determine QAT's final performance**
  > During the experiment, I found that improper choice of learning rate of QAT would make the performance of it worse than simple PTQ. After many adjustments to the learning rate, QAT's final performance can surpass PTQ. The empirical law is: When we fine-tune a full precision model, use lower learning rate for lower bits quantized model and higher learning rate for higher bits quantized model(anyway, the learning rate for QAT is much lower than full precision model training setting).

- **Batch-Normalization layers lower the accuracy of quantized model**
  > From the above experiment results we can find that the quantized models with folded Conv-BN-RELU layers actually have worse performance. One possible explanation is: We fix the folded Conv-BN-RELU layers' weights and bias using **exponential average running mean and running std** obtained during quantization aware training, however, the **actual folded weights and bias corresponds to the actual inputs during inference** , and this operation brings estimated error. Since the mean and std of inputs may be more **sensitive** than the max and min value of them(used to calculate S and Z), the final results of quantized models with Conv-BN-RELU layers are worse.

- **Optimizer affects QAT's performance**
  > At first, I choose to use Adam for quantization aware training. However, its performance is worse than use SGD with momentum. Currently, I cannot figure out the reason behind this phenomenon.

## TODO list

- Train a model from scratch with quantization aware training
- Read more paper like LSQ, LSQ+
- Try to read the source code of TensorFlow Lite and reproduce it from the underlying code