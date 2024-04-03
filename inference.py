import torch
import torch.nn as nn
from pipnet.test import eval_pipnet, get_thresholds, eval_ood
from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, save_args, get_optimizer_nn
import random
import numpy as np
from util.data import get_dataloaders
from PIL import Image
import torchvision.transforms as transforms
import torchvision


if __name__ == "__main__":

    out_dir = './test_images/out.txt'

    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load an image
    image = Image.open('./test_images/' + args.image_filename).convert('RGB')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(224, 224)),
                            transforms.ToTensor(),
                            normalize
                        ])
    image = transform_no_augment(image).unsqueeze(0)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(out_dir, 'w') as f:
        f.write(f"Device: {device}\n\n")
    image = image.to(device)

    # Obtain the dataset and dataloaders
    # trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)

    with open('classes.txt', 'r') as f:
        classes = f.readlines()

    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
    
    # Create a PIP-Net
    net = PIPNet(num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)    

    checkpoint = torch.load('dog_66percent_newloss', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'],strict=True)
    with open(out_dir, 'a') as f:
        f.write("Pretrained network loaded\n\n")

    net.eval()
    with torch.no_grad():
        proto_features, pooled, out = net(image, inference=True)
        proto_features = proto_features.squeeze(0)
        pooled = pooled.squeeze(0)
        out = out.squeeze(0)
        # for i, score in enumerate(out):
        #     if score < 10:
        #         out[i] = 0

        with open(out_dir, 'a') as f:
            f.write(f'proto_features:\n{proto_features}\n\n')
            f.write(f'Shape of proto_features:\n{proto_features.shape}\n\n')
            f.write(f'pooled:\n{pooled}\n\n')
            f.write(f'Shape of pooled:\n{pooled.shape}\n\n')
            f.write(f'out:\n{out}\n\n')
            f.write(f'Shape of out:\n{out.shape}\n\n')


        max_out_score, ys_pred = torch.max(out, dim=0)
        with open(out_dir, 'a') as f:
            f.write(f'max_out_score:\n{max_out_score}\n\n')
            f.write(f'ys_pred:\n{ys_pred}\n\n')
        if max_out_score < 7:
            with open(out_dir, 'a') as f:
                f.write('I HAVE NOT SEEN THIS BEFORE!\n')
            print('\nI HAVE NOT SEEN THIS BEFORE!\n')
        else:
            with open(out_dir, 'a') as f:
                f.write(f'\nThe input image is a {classes[ys_pred.item()]}')
            print(f'\nThe input image is a {classes[ys_pred.item()]}')