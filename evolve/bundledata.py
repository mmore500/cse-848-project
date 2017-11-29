import torch
import torchvision

def bundledata(listofpops):
    listofinds = [ind for pop in listofpops for ind in pop]
    bundled_testdata = torch.Tensor(listofinds)

    print(bundled_testdata.shape)

    torch.save(bundled_testdata, "../data/traindata.pt")
