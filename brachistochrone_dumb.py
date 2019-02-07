import torch
import torch.autograd
import matplotlib.pyplot as plt

xs = torch.linspace(0,1,25)
ys = torch.zeros(25, requires_grad=True)
print(ys)

vs = [1. - y for y in ys] + [1.]
diffs = [1 - ys[0]] +  [y - y_last for (y, y_last) in zip(ys[1:], ys[:-1])] + [ys[-1]]

slopes = diffs/(xs[1]-xs[0])
slowdown = 1./torch.cos(torch.atan(slopes))


ts = torch.cumsum(vs[1:] * slowdown, dim=0)
t = ts[-1]
t.backward()
print(ys.grad)

plt.plot(xs.numpy(), ys.detach().numpy())
plt.plot(xs.numpy()[1:], ts.detach().numpy())
plt.show()
