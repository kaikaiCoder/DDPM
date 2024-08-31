import matplotlib.pyplot as plt
import requests
import torchvision.transforms as transforms
from PIL import Image
import torch


def preprocess_img(img):
  transform =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64)), # resize to 64x64
    transforms.Lambda(lambda x: x * 2 - 1), # normalize to [-1, 1]
  ])
  return transform(img)

def reverse_img(tensor):
    transform = transforms.Compose([
      transforms.Lambda(lambda x: (x + 1) * 255 / 2),  # 反向归一化到 [0, 255]
      transforms.Lambda(lambda x: x.byte()),  # [1,H,W,C] to [H,W,C] and cast to uint8
      transforms.ToPILImage()  # convert to PIL image
    ])
    return transform(tensor)

def q_sample(x0,t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod):
  noise = torch.randn_like(x0)
  return sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * noise

def get_noise_img(img,t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod):
  img = preprocess_img(img)
  x_noised = q_sample(img,t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod)
  return reverse_img(x_noised)


def show_noise_imgs(x0,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod):
  fig = plt.figure(figsize=(8,8))
  columns = 5
  rows = 4
  for i in range(1,rows * columns +1):
    t = 10*i-1
    img = get_noise_img(x0,t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod)
    fig.add_subplot(rows,columns,i).set_title(f"t={t+1}")
    plt.imshow(img)
    plt.axis('off')
  plt.show()

T = 1000
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alphas_comprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_comprod = torch.sqrt(alphas_comprod)
sqrt_one_minus_alphas_comprod = torch.sqrt(1 - alphas_comprod)

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# original_image = Image.open(requests.get(url, stream=True).raw)
# transform = transforms.ToTensor()
# original_image = transform(original_image)
# print(original_image.shape)
# show_noise_imgs(original_image,sqrt_alphas_comprod,sqrt_one_minus_alphas_comprod)


class GaussianDiffusion:
  def __init__(self,beta_start,beta_end,n_step,clip_min=-1.0,clip_max=1.0):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.n_step = n_step
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.betas = torch.linspace(beta_start,beta_end,n_step)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = torch.log(1.0 / self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / (self.alphas_cumprod - 1))

    alphas_cumprod_prev  = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
    self.posterior_variance = torch.sqrt(self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
    self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-6))
    self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) /(1.0-self.alphas_cumprod)
    self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

  def predict_start_from_noise(self,xt,t,noise):
    '''
    从噪声还原 x0
    x0 = sqrt_recip_alphas_cumprod_t * xt + sqrt_recipm1_alphas_cumprod_t * noise
    Args:
      xt: torch.Tensor, shape [B, C, H, W]
      t: int, time step
      noise: torch.Tensor, shape [B, C, H, W]
    Returns:
      samples: torch.Tensor, shape [B, C, H, W]
    '''
    B= xt.shape[0]
    sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(B,1,1,1)
    sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(B,1,1,1)
    return sqrt_recip_alphas_cumprod_t * xt + sqrt_recipm1_alphas_cumprod_t * noise


  def q_sample(self,x0,t,noise):
    '''
    前向添加噪声过程,根据公式x_t = sqrt_alphas_comprod_t * x_0 + sqrt_one_minus_alphas_comprod_t * noise
    Args:
      x0: torch.Tensor, shape [B, C, H, W]
      t: int, time step
      noise: torch.Tensor, shape [B, C, H, W]
    Returns:
      diffused samples at t: torch.Tensor, shape [B, C, H, W]
    '''
    B = x0.shape[0]
    sqrt_alphas_comprod_t = self.sqrt_alphas_cumprod[t].view(B,1,1,1)
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B,1,1,1)
    return sqrt_alphas_comprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

  def q_posterior(self,x0,xt,t):
    '''
    计算 q(x_{t-1}|x_t,x_0) 均值和方差
    Args:
      x0: torch.Tensor, shape [B, C, H, W]
      xt: torch.Tensor, shape [B, C, H, W]
      t: int, time step
    Returns:
      posterior mean and log variance: torch.Tensor, torch.Tensor, shape [B, C, H, W]
    '''
    self.posterior_mean = self.posterior_mean_coef1[t] * x0 + self.posterior_mean_coef2[t] * xt
    return self.posterior_mean, self.posterior_variance[t], self.posterior_log_variance_clipped[t]

  def p_mean_variance(self,pred_noise,x_t,t,clip=True):
    '''
    p(x_t|x_{t-1})
    获得通过模型预测得到的均值和方差
    Args:
      pred_noise: torch.Tensor, shape [B, C, H, W]
      x_t: torch.Tensor, shape [B, C, H, W]
      t: int, time step
      clip: bool, 是否裁剪
    Returns:
      mean and variance: torch.Tensor, torch.Tensor, shape [B, C, H, W]
    '''
    B = x_t.shape[0]
    x0 = self.predict_start_from_noise(x_t,t,pred_noise) # 从噪声还原 x0
    if clip:
      x0 = x0.clamp(self.clip_min,self.clip_max)
    model_mean,model_variance,model_log_variance = self.q_posterior(x0,x_t,t)
    return model_mean,model_variance,model_log_variance
  
  def p_sample(self,pred_noise,x_t,t):
    '''
    通过模型预测得到的均值和方差,采样
    Args:
      pred_noise: torch.Tensor, shape [B, C, H, W]
      x_t: torch.Tensor, shape [B, C, H, W]
      t: int, time step
    Returns:
      samples: torch.Tensor, shape [B, C, H, W]
    '''
    model_mean,model_variance,model_log_variance = self.p_mean_variance(pred_noise,x_t,t)
    noise = torch.randn_like(x_t)
    # Create a mask to avoid adding noise when t == 0
    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
    return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise