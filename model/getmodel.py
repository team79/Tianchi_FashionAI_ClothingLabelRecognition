import pretrainedmodels
from ulit.ulits import *


def models_print():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    print(model_names)


poolsize = 1


def get_model(args):
    models_print()
    print("=> creating model '{}'".format(args.arch))

    model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    if args.arch == 'resnet152':
        model.avgpool = nn.AdaptiveAvgPool2d(poolsize)
        input_size = 448
    elif args.arch == 'nasnetalarge':
        model.avg_pool = nn.AdaptiveAvgPool2d(poolsize)
        input_size = 427
    elif args.arch == 'inceptionv4':
        model.avgpool = nn.AdaptiveAvgPool2d(poolsize)
        input_size = 491
    elif args.arch == 'inceptionresnetv2':
        model.avgpool_1a = nn.AdaptiveAvgPool2d(poolsize)
        input_size = 491
    elif args.arch == 'se_resnext101_32x4d':
        model.avg_pool = nn.AdaptiveAvgPool2d(poolsize)
        input_size = 448

    model = modelfinetune(model, task_dict[args.task], args)

    return model, input_size


class modelfinetune(nn.Module):
    def __init__(self, model, num_class, args, droup=0.5):
        super(modelfinetune, self).__init__()
        self.model_features = model
        self.drop = nn.Dropout(0.25)
        self.linear = nn.Linear(1000, num_class)

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        elif hasattr(self, 'model_features'):  # compatible with early verion
            x = self.model_features(x)
            x = x.view(x.size(0), -1)
            x = self.drop(x)
            x = self.linear(x)
        elif hasattr(self, 'model'):  # nasnet
            x = self.model(x)
        else:
            x = self.basemodel(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            if hasattr(self, 'drop'):
                x = self.drop(x)
            x = self.classifier(x)
        return x
