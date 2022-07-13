# %%
import math
import torch

# %%
model=torch.load('model.pth')
model.eval()

# %%


# %%
test=torch.ones(600,1)
test_x=torch.linspace(-3,3,600)
test[:,0]=test_x
true_y=func(test_x)
test_y=model(test).clone().detach()
print(test_y)
error=abs(test_y/true_y-1)
print("Max error =",torch.max(error).item()*100,"%")
plt.subplot(211)
plt.plot(test_x,true_y,label='Reference')
plt.plot(test_x,test_y,label='Fitting')
plt.ylabel("y")
plt.title("Comparison")
plt.legend()
plt.subplot(212)
plt.plot(test_x,error,label='Error')
plt.xlabel("x")
plt.ylabel("Error (*100%)")
plt.title("Error")
plt.show()
