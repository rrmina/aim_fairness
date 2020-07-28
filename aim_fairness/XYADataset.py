import torch

# XYADataset
# This is torch.utils.data.Dataset for common fairness datasets 
# which is composed of (X,y,a) -> (Features, Labels, Sensitive Attribute)
class XYADataset(torch.utils.data.Dataset):
    def __init__(self, x, y, a):
        super(XYADataset, self).__init__()
        assert x.shape[0] == y.shape[0], "Features and Labels are expected to have same number of instances"
        assert y.shape[0] == a.shape[0], "Labels and Sensitive Attributes are expected to have same number of instances"
        self.x = x
        self.y = y
        self.a = a
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.a[index]
    
    def __len__(self):
        return self.x.shape[0]