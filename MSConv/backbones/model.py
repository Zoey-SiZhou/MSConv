import torch.nn as nn
import torch
from functools import reduce
from timm.models.layers import DropPath, trunc_normal_
from timm.models import register_model

__all__ = ['msnet26', 'msnet50', 'msnet101']


class MSConv(nn.Module):
    def __init__(self, features, ratio=1, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(MSConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        self.hidden_dim = ratio * features
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, self.hidden_dim, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          groups=G,
                          bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.PReLU()
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(self.hidden_dim, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.PReLU()
                                )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Sequential(
                nn.Conv2d(d, self.hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.PReLU()
            ))
        self.fc_d = nn.Sequential(nn.Conv2d(self.hidden_dim, features, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(features),
                                  nn.PReLU()
                                  )

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.hidden_dim, feats.shape[2], feats.shape[3])

        feats_U = feats[:, 0] * feats[:, 1]
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vector_1 = attention_vectors[0]
        attention_vector_2 = attention_vectors[1]
        c = torch.sigmoid((attention_vector_1 - attention_vector_2) + 1e-6)

        U3 = feats[:, 0] - feats[:, 1]
        U3_c = U3 * c
        final_output = U3_c + feats[:, 1]
        final_output = self.fc_d(final_output)

        return final_output


class MSUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(MSUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.PReLU()
        )

        self.conv2_msconv = MSConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
        )

        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )

        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2_msconv(out)
        out = self.conv3(out)

        return self.prelu(out + self.shortcut(residual))


class MSNet(nn.Module):
    def __init__(self, class_num, nums_block_list=[3, 4, 6, 3], strides_list=[1, 2, 2, 2], dropout_prob=0.5):
        super(MSNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])

        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, class_num)

    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers = [MSUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(MSUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


def msnet26(nums_class=512):
    return MSNet(nums_class, [2, 2, 2, 2])


def msnet50(nums_class=512):
    return MSNet(nums_class, [3, 4, 6, 3])


def msnet101(nums_class=512):
    return MSNet(nums_class, [3, 4, 23, 3])
