import torch
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

device = torch.device("cpu")

# >>>>tutorial >>>>>>

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a + 2 # adds over the entire tensor
c=torch.asarray(b) # converts array to tensor
print(c)

#####

a = np.array([1,2,3])
b=torch.from_numpy(a)
print(b)

####

print(torch.zeros((2,3)))

a = torch.tensor([[1,2],[3,4]])
print(torch.zeros_like(a)) # the input must also be a tensor
# ditto same for ones and one_like

####

print(torch.arange(0,9,2))

#####

print(torch.linspace(0,9,2)) # inclusive

####

print(torch.full((2, 3), 3.141592))
print(torch.full_like(a, 3.141592))
# ditto same for empty and empty_like

#####

a = torch.arange(4)
print(torch.reshape(a, (2, 2)))

###3

x = torch.randn(2, 3)
print(torch.cat((x, x, x), 0)) # torch.concat or torch.concatenate
print(torch.cat((x, x, x), 1))

#####

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.column_stack((a, b))) #

a = torch.arange(5)
b = torch.arange(10).reshape(5, 2)
print(torch.column_stack((a, b, b)))

#####

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.hstack((a,b)))
a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[4],[5],[6]])
print(torch.hstack((a,b)))

####

x = torch.randn(2, 3)
print(torch.stack((x, x))) # same as print(torch.stack((x, x), dim=0)
print(torch.stack((x, x), dim=1))
print(torch.stack((x, x), dim=2))

####

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.vstack((a,b)))
a = torch.tensor([[1],[2],[3]])
b = torch.tensor([[4],[5],[6]])
print(torch.vstack((a,b)))