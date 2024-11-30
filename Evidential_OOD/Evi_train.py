import os
import torch
import argparse
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import validate
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
from utils.anom_utils import ToLabel
from model.classifiersimple import *

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits shape: (N, D * 2) - where the last dimension represents concatenated (a, b)
        # targets shape: (N, D) - binary indicators for each class (one-hot encoded)

        # Reshape logits to get a and b
        # a and b will have shape (N, D)
        a = logits[:, :logits.shape[1] // 2]  # First half corresponds to a
        b = logits[:, logits.shape[1] // 2:]   # Second half corresponds to b

        # Calculate p as a / (a + b) for each class
        p = a / (a + b)

        # Compute cross-entropy: -[target * log(p) + (1 - target) * log(1 - p)]
        cross_entropy_loss = -(
            targets * torch.log(p) + (1 - targets) * torch.log(1 - p)
        )

        # Sum over labels for each instance
        loss = cross_entropy_loss.sum(dim=1)  # Sum over the labels for each instance

        # Apply reduction
        if self.reduction == 'sum':
            return loss.sum()  # Sum all losses for the batch
        elif self.reduction == 'mean':
            return loss.mean()  # Average loss for the batch
        else:
            return loss  # 'none' reduction, return individual losses for each instance


def train():
    args.save_dir += args.dataset + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomResizedCrop((256, 256), scale=(0.5, 2.0)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    label_transform = torchvision.transforms.Compose([
            ToLabel(),
        ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    if args.dataset == "pascal":
        loader = pascalVOCLoader(
                                 "./datasets/pascal/",
                                 img_transform = img_transform,
                                 label_transform = label_transform)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform,
                                   label_transform=label_transform)

    elif args.dataset == "coco":
        loader = cocoloader(".\datasets\coco\\", split="multi-label-train2014",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = cocoloader(".\datasets\\coco\\", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)

    elif args.dataset == "nus-wide":
        loader = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                                 img_transform=val_transform,
                                 label_transform=label_transform)
    else:
        raise AssertionError

    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    print("number of images = ", len(loader))
    print("number of classes = ", args.n_classes, " architecture used = ", args.arch)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model= nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, args.n_classes)
    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        #clsfier = clssimp(1024, 2*args.n_classes)
        clsfier = LeNetAdapter(1024, 2*args.n_classes)

    model = model.cuda()
    clsfier = clsfier.cuda()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        clsfier = nn.DataParallel(clsfier)

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.l_rate/10},{'params': clsfier.parameters()}], lr=args.l_rate)

    if args.load:
        model.load_state_dict(torch.load(args.save_dir + args.arch + ".pth"))
        clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clsfier' + ".pth"))
        print("Model loaded!")

    criterion = CustomCrossEntropyLoss()
    # bceloss = nn.BCEWithLogitsLoss()
    model.train()
    clsfier.train()

    best_mAP = 0.0  # Initialize the best evaluation score
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().float())

            optimizer.zero_grad()

            outputs = model(images)
            # outputs = F.relu(outputs, inplace=True)
            outputs = clsfier(outputs)
            outputs = torch.nn.functional.softplus(outputs) + 1e-5

            # outputs = model(images)
            # outputs = clsfier(outputs)
            # loss = bceloss(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        mAP = validate.evi_validate(args, model, clsfier, val_loader)
        print("Epoch [%d/%d] Loss: %.4f mAP: %.4f" % (epoch, args.n_epoch, loss.data, mAP))
        if mAP > best_mAP:
            best_mAP = mAP
            print(f"New best mAP: {best_mAP:.4f}, saving model...")
            torch.save(model.state_dict(), args.save_dir + args.arch + "_evi_" + ".pth")
            torch.save(clsfier.state_dict(), args.save_dir + args.arch + '_evi_clsfier' + ".pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='# of the epochs')
    parser.add_argument('--n_classes', type=int, default=20,
                        help='# of classes')
    parser.add_argument('--batch_size', type=int, default=110,
                        help='Batch Size')
    # batch_size 320 for resenet101
    parser.add_argument('--l_rate', type=float, default=1e-4,
                        help='Learning Rate')

    #save and load
    parser.add_argument('--load', action='store_true', help='Whether to load models')
    parser.add_argument('--save_dir', type=str, default=".\saved_models\evi\\",
                        help='Path to save models')

    args = parser.parse_args()
    train()

