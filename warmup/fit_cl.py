import torch
import numpy as np
import matplotlib.pyplot as plt

#fit di curva CL alpha con tensori e autograd
#1.6449
alpha_deg = torch.linspace(-5,20,25)
alpha_rad = torch.deg2rad(alpha_deg)
cl_lin = torch.pi*2*alpha_rad
cl_clamped = torch.clamp(cl_lin, max = 1.6449)
seed = torch.manual_seed(0)
cl_data = 0.05*torch.randn_like(cl_clamped) + cl_clamped

#you need 2 scalar tensors
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

N = 10000
lr=0.01
loss_hist = []
w_hist = []
b_hist = []
for epoch in range(N):
    pred = w*alpha_rad + b
    loss = torch.mean((pred-cl_data)**2)
    loss.backward()
    loss_hist.append(loss.item())
    w_hist.append(w.item())
    b_hist.append(b.item())
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
        w.grad = None
        b.grad.zero_() #operazione in place equivalente
        pred_vec = w*alpha_rad + b


fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].plot(loss_hist)
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("loss")
axs[0].set_yscale("log")

axs[1].plot(w_hist)
axs[1].plot(b_hist)
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("weights")

fig2, ax2 = plt.subplots()
ax2.plot(alpha_rad, cl_data)
ax2.plot(alpha_rad, pred_vec)


plt.show()

