import torch
from model import LSTMnet
from train import batch_scale, make_map
import matplotlib.pyplot as plt
from Dataloader import test_loader 

def evaluate(model, test_loader, label_maps):
    # Nghiệm chứng và đánh giá 
    test_features, test_labels = batch_scale(next(iter(test_loader)))

    val = model(test_features)
    _ , pred = torch.max(val.data, 1)
    acc = (100 * torch.sum(test_labels==pred) / len(test_labels))
    print(f'Accuracy of the network {acc} %')

    idx = 30
    val = model(test_features[:idx])
    _ , pred = torch.max(val.data, 1)

    fig = plt.figure(figsize=(15,15))
    for i in range(0,idx):
        img = test_features[i].squeeze()
        fig.add_subplot(5,6,i+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        plt.title(label_maps[int(pred[i])] + ' <-> ' + label_maps[int(test_labels[i])])
    plt.show()
    
if __name__ == '__main__':
    label_maps , class_num = make_map()
    input_dim = 28
    hidden_dim = 128
    layer_dim = 1
    output_dim = class_num
    model = LSTMnet(input_dim, hidden_dim, layer_dim, output_dim)
    model.load_state_dict(torch.load('Net.pth'))
    evaluate(model, test_loader, label_maps)