from typing import List

import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from onai.ml.tools.file import cached_fs_open

BN_MOMENTUM = 0.1

MU = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [512, 512]


MODEL_WEIGHT_URL = (
    "s3://oaknorth-staging-non-confidential-ml-artefacts/tablestructure/0.0.2/model.pth"
)
ONNXM_WEIGHT_URL = "s3://oaknorth-staging-non-confidential-ml-artefacts/tablestructure/0.0.0/model.onnx"


class CenterNetOnnx:
    def __init__(self):
        with cached_fs_open(ONNXM_WEIGHT_URL) as fin:
            self.session = ort.InferenceSession(fin.read())

    def __call__(self, input):
        input = input.to(torch.float32).numpy()
        x = self.session.run(None, {"input.1": input})
        return {"heatmap": torch.tensor(x[0])}


class CenterNet(nn.Module):
    def __init__(self, device):
        if "cuda" in str(device) and not torch.cuda.is_available():
            raise RuntimeError(f"Cuda is not available for device {device}")
        super().__init__()
        m = models.resnet34(pretrained=False)
        self.backbone = models._utils.IntermediateLayerGetter(
            m, {"layer4": "out", "layer3": "aux"}
        )
        self.backbone.out_planes = 512
        self.deconv = ResnetDeconv(
            self.backbone.out_planes,
            deconv_kernels=[4, 4, 4],
            deconv_planes=[256, 256, 256],
            bias=[False],
        )

        self.heatmap = OutHead(self.deconv.out_planes, self.deconv.out_planes, 1)
        self.to(device)
        with cached_fs_open(MODEL_WEIGHT_URL) as fin:
            weights = torch.load(fin, map_location=device)["model"]
        self.load_state_dict(weights)
        del weights
        self.backbone.eval()
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        output_size = x.shape[-2:]
        x = self.backbone(x)["out"]
        x = self.deconv(x)
        output = {}
        heatmap = torch.sigmoid(self.heatmap(x))
        output["heatmap"] = self._upsample_heatmap(heatmap, output_size)
        return output

    @staticmethod
    def _upsample_heatmap(heatmap, size):
        return F.interpolate(
            heatmap, size=size, mode="bilinear", align_corners=False
        ).squeeze(1)


class OutHead(nn.Module):
    def __init__(self, in_planes, inner_planes, out_planes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_planes, out_planes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class ResnetDeconv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        deconv_kernels: List[int],
        deconv_planes: List[int],
        bias=False,
    ):
        assert len(deconv_kernels) == len(deconv_planes)

        super().__init__()
        layers = []
        for kernel, planes in zip(deconv_kernels, deconv_planes):
            kernel, in_padding, out_padding = self._get_deconv_params(kernel)

            layers.append(
                DeconvBlock(
                    in_planes=in_planes,
                    out_planes=planes,
                    kernel=kernel,
                    in_padding=in_padding,
                    out_padding=out_padding,
                    stride=2,
                    bias=True,
                )
            )
            in_planes = planes

        self.out_planes = planes
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _get_deconv_params(deconv_kernel):
        if deconv_kernel == 4:
            in_padding = 1
            out_padding = 0
        elif deconv_kernel == 3:
            in_padding = 1
            out_padding = 1
        elif deconv_kernel == 2:
            in_padding = 0
            out_padding = 0

        return deconv_kernel, in_padding, out_padding


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel,
        in_padding,
        out_padding,
        stride=2,
        bias=True,
    ):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel,
            stride=2,
            padding=in_padding,
            output_padding=out_padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.out_planes = out_planes

    def forward(self, x):
        return self.relu(self.bn(self.conv_t(x)))
