import torch
import matplotlib.pyplot as plt
from Dataloader import test_loader, test_data
from model import Net

def make_map():
    label_maps = dict()
    class_num = len(test_data.classes)
    label_maps = {x:test_data.classes[x] for x in range(class_num)}
    
    return label_maps, class_num

def evaluate(model, test_loader, label_maps):

    test_features, test_labels = next(iter(test_loader))
    val = model(test_features)
    _, pred = torch.max(val.data, 1)
    acc = (100 * torch.sum(test_labels==pred) / len(test_labels))
    print(f'Accuracy of the network {acc} %')

    idx = 30
    val = model(test_features[:idx])
    _, pred = torch.max(val.data, 1)
    fig = plt.figure(figsize=(15,15))
    
    for i in range(0,idx):
        img = test_features[i].squeeze()
        fig.add_subplot(5,6,i+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        plt.title(label_maps[int(pred[i])] + ' <-> ' + label_maps[int(test_labels[i])])
    plt.show()
if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('Net.pth'))
    label_maps, _ = make_map()
    evaluate(model, test_loader, label_maps)