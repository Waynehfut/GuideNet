# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       cnn_finetune on workflowpredict
   Description:
   Author:          HAO
   Date:            2019/10/14
   Create by:       PyCharm
   Check status:    https://waynehfut.com
-------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, cnn_embed_dim=300, hidden_lstm_layers=3,
                 hidden_lstm=256, hidden_fc_dim=128, categories=7):
        """Load pretrained resnet50 and replace top fc layer"""
        super(CNNLSTM, self).__init__()
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.drop_p = drop_p

        self.lstm_input_size = cnn_embed_dim
        self.hidden_lstm_layers = hidden_lstm_layers
        self.hidden_lstm = hidden_lstm
        self.hidden_fc_dim = hidden_fc_dim
        self.categories = categories

        resnet = models.resnet50()
        modules = list(resnet.children())[:-1]  # remove last fc layer
        # * can cover the list to a element
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)
        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_lstm,
            num_layers=hidden_lstm_layers,
            batch_first=True
        )

        self.fc4 = nn.Linear(self.hidden_lstm, self.hidden_fc_dim)
        self.fc5 = nn.Linear(self.hidden_fc_dim, self.categories)

    def forward(self, x_3d):
        # embedding the cnn feature
        # x_3d shape: [batch_size, sample_duration, channel, width, height]
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():  # pretrained no grad
                x = self.resnet(x_3d[:, t, :, :, :])  # Resnet
                x = x.view(x.size(0), -1)  # flatten output

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
        # swap time and sample
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        self.LSTM.flatten_parameters()
        LSTM_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        # FC layers
        x = self.fc4(LSTM_out[:, -1, :])  # select last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p)
        x = self.fc5(x)
        return x


class CurrentPredict(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, cnn_embed_dim=300, hidden_lstm_layers=3,
                 hidden_lstm=256, hidden_fc_dim=128, softmax_dim=1, lambda_1=0.8, categories=7):
        super(CurrentPredict, self).__init__()
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.drop_p = drop_p

        self.lstm_input_size = cnn_embed_dim
        self.hidden_lstm_layers = hidden_lstm_layers
        self.hidden_lstm = hidden_lstm
        self.hidden_fc_dim = hidden_fc_dim
        self.softmax_dim = softmax_dim
        self.drop_p = drop_p
        self.categories = categories

        resnet = models.resnet50()
        modules = list(resnet.children())[:-1]  # remove last fc layer
        # * can cover the list to a element
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.1)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_lstm,
            num_layers=hidden_lstm_layers,
            batch_first=True
        )
        self.fc4 = nn.Linear(self.hidden_lstm, self.hidden_fc_dim)
        # Time 2 for previous input and future predict

        self.lambda_1 = torch.nn.Parameter(torch.tensor([lambda_1]))
        self.softmax = nn.Softmax(dim=1)
        self.fc5 = nn.Linear(self.hidden_fc_dim * 2, self.categories * 2)

    def forward(self, x_3d, previous_info):
        """
        Current forward
        :param x_3d: current 3d sequential data
        :param previous_info: previous extract 128dim feature
        :return: predict result, trained predict lambda
        """
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():  # pretrained no grad
                x = self.resnet(x_3d[:, t, :, :, :])  # Resnet
                x = x.view(x.size(0), -1)  # flatten output

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.fc2(self.bn2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
        # swap time and sample
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        self.LSTM.flatten_parameters()
        LSTM_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        # FC layers
        x = self.fc4(LSTM_out[:, -1, :])  # select last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        # previous guide merge
        previous_info = previous_info * self.lambda_1
        x = torch.stack((previous_info, x), dim=self.softmax_dim)
        x = x.view(-1, 256)
        x = self.softmax(x)
        x = self.fc5(x)  # dim is 14
        return x, self.lambda_1


class PreviousGuide(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, cnn_embed_dim=300, hidden_lstm_layers=3,
                 hidden_lstm=256, hidden_fc_dim=128, categories=7):
        super(PreviousGuide, self).__init__()
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.drop_p = drop_p

        self.lstm_input_size = cnn_embed_dim
        self.hidden_lstm_layers = hidden_lstm_layers
        self.hidden_lstm = hidden_lstm
        self.hidden_fc_dim = hidden_fc_dim
        self.categories = categories

        resnet = models.resnet50()
        modules = list(resnet.children())[:-1]  # remove last fc layer
        # * can cover the list to a element
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.1)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_lstm,
            num_layers=hidden_lstm_layers,
            batch_first=True
        )
        self.fc4 = nn.Linear(self.hidden_lstm, self.hidden_fc_dim)
        # self.softmax=nn.Softmax(dim=1)
        # self.fc5 = nn.Linear(self.hidden_fc_dim, self.categories)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():  # pretrained no grad
                m = x_3d[:, t, :, :, :]
                x = self.resnet(x_3d[:, t, :, :, :])  # Resnet
                x = x.view(x.size(0), -1)  # flatten output

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
        # swap time and sample
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        self.LSTM.flatten_parameters()
        LSTM_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        # FC layers
        x = self.fc4(LSTM_out[:, -1, :])  # select last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        # previous guide merge

        # x = torch.stack(x, dim=1)
        # x = self.softmax(x)
        # x = self.fc5(x)
        return x


def initialize_cnn_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize these variables which will be set in this if statement. Each of these
    variables is model specific.

    Reference: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    :param model_name: [resnet, vgg, squeezenet, densenet, inception]
    :param num_classes: num
    :param feature_extract: Flag for feature extracting. When False, we finetune the whole model,
                            when True we only update the reshaped layer params
    :param use_pretrained: finetune
    :return: model and inputsize
    """

    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 512

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 512

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
