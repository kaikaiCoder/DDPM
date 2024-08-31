# DDPM (Denoising Diffusion Probabilistic Models) Implementation

本项目实现了 DDPM（Denoising Diffusion Probabilistic Models），一种用于生成模型的最新方法。DDPM 通过逐步添加噪声并反向去噪声来生成高质量的样本，广泛应用于图像生成任务。

## 公式推导

### 前向过程（Forward Process）

在 DDPM 中，前向过程逐步添加噪声到数据 $x_0$，生成一系列 $x_1, x_2, \dots, x_T$。这个过程可以表示为：

$$ q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}) $$

其中，$\beta_t$ 是定义噪声强度的常数。

### 反向过程（Reverse Process）

反向过程尝试从噪声 $x_T$ 逐步去噪还原数据，建模 $p(x_{t-1} | x_t)$。用神经网络参数化这个过程：

$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma^2_t \mathbf{I}) $$

