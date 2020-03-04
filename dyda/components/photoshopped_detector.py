import pdb
import cv2
import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from dyda_utils import lab_tools
from PIL import Image
from dyda.core import image_processor_base
from dyda.core import detector_base
torch.backends.cudnn.benchmark = True
"""This componet is almost modfied from
   https://github.com/PeterWang512/FALdetector"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = self.conv3x3(inplanes, planes, stride,
                                  padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes,
                                  padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

    def conv3x3(self, in_planes, out_planes, stride=1, padding=1, dilation=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=padding, bias=False, dilation=dilation)


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(
                BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(
                BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                          bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(
                channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(
                channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4,
                             new_level=False)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6],
                                 dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7],
                                 dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x


class DRNSeg(nn.Module):
    def __init__(self, classes, pretrained_drn=False,
                 pretrained_model=None, use_torch_up=False):
        super(DRNSeg, self).__init__()

        model = self.drn_c_26(pretrained=pretrained_drn)
        self.base = nn.Sequential(*list(model.children())[:-2])
        if pretrained_model:
            self.load_pretrained(pretrained_model)

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)

        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def drn_c_26(self, pretrained=False, **kwargs):
        webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'
        model_urls = webroot + 'drn_c_26-ddedf421.pth'

        model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls))
        return model

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

    def load_pretrained(self, pretrained_model):
        print("loading the pretrained drn model from %s" % pretrained_model)
        state_dict = torch.load(pretrained_model, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # filter out unnecessary keys
        pretrained_dict = state_dict['model']
        pretrained_dict = {k[5:]: v for k, v in pretrained_dict.items()
                           if k.split('.')[0] == 'base'}

        # load the pretrained state dict
        self.base.load_state_dict(pretrained_dict)

    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class DRNSub(nn.Module):
    def __init__(self, num_classes, pretrained_model=None, fix_base=False):
        super(DRNSub, self).__init__()

        drnseg = DRNSeg(2)
        if pretrained_model:
            print(
                "loading the pretrained drn model from %s" %
                pretrained_model)
            state_dict = torch.load(pretrained_model, map_location='cpu')
            drnseg.load_state_dict(state_dict['model'])

        self.base = drnseg.base
        if fix_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PhotoshoppedFaceDetector(detector_base.DetectorBase):
    """ Detector photoshopped face

        input: IMAGE_OF_FACE, or LIST_OF_IMAGE_OF_FACE
             => np.array, or [np.array, np.array, ....]

        output: IMAGE_OF_FACE, or LIST_OF_IMAGE_OF_FACE
             => np.array, or [np.array, np.array, ....]

        @param model_path: the path of torch model .pth

        @param gpu_id: the id of gpu wanted to use

        @param conf_thre: the threshold of confidence. If
             "photoshopped_prob" is bigger than this value,
             in verifications return "Mismatched."

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component.

        """

        super(PhotoshoppedFaceDetector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.model_path = self.param["model_path"]

        if "gpu_id" in self.param.keys():
            gpu_id = self.param["gpu_id"]
        else:
            gpu_id = -1

        if torch.cuda.is_available() and gpu_id != -1:
            self.device = 'cuda:{}'.format(gpu_id)
        else:
            self.device = 'cpu'

        if "conf_thre" in self.param.keys():
            self.conf_thre = self.param["conf_thre"]
        else:
            self.conf_thre = 0.3

        self.model = DRNSub(1)
        state_dict = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)
        self.model.device = self.device
        self.model.eval()

        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        self.input_data = self.uniform_input()

        for input_img in self.input_data:

            im_w = input_img.shape[1]
            im_h = input_img.shape[0]

            input_img = Image.fromarray(cv2.cvtColor(input_img,
                                                     cv2.COLOR_BGR2RGB))
            face = self.resize_shorter_side(input_img, 400)[0]
            face_tens = self.tf(face).to(self.device)

            # Prediction
            with torch.no_grad():
                prob = self.model(face_tens.unsqueeze(0)
                                  )[0].sigmoid().cpu().item()
            results = lab_tools.output_pred_detection("", "")
            results["verifications"] = []
            results["verifications"].append({
                "type": "photoshopped_face",
                "confidence": prob,
                "result": "Passed" if prob <= self.conf_thre else "Mismatched"
            })
            self.results.append(results)
        self.uniform_output()

    def resize_shorter_side(self, img, min_length):
        """
        Resize the shorter side of img to min_length while
        preserving the aspect ratio.
        """
        ow, oh = img.size
        mult = 8
        if ow < oh:
            if ow == min_length and oh % mult == 0:
                return img, (ow, oh)
            w = min_length
            h = int(min_length * oh / ow)
        else:
            if oh == min_length and ow % mult == 0:
                return img, (ow, oh)
            h = min_length
            w = int(min_length * ow / oh)
        return img.resize((w, h), Image.BICUBIC), (w, h)


class PhotoshoppedFaceReverser(image_processor_base.ImageProcessorBase):
    """ Reverse photoshopped face

        input: IMAGE_OF_FACE, or LIST_OF_IMAGE_OF_FACE
             => np.array, or [np.array, np.array, ....]

        output: IMAGE_OF_FACE, or LIST_OF_IMAGE_OF_FACE
             => np.array, or [np.array, np.array, ....]

        @param model_path: the path of torch model .pth

        @param gpu_id: the id of gpu wanted to use

        @param return_heatmap: does return the heatmap of
         photoshop trace
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component.

        """

        super(PhotoshoppedFaceReverser, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.model_path = self.param["model_path"]

        if "gpu_id" in self.param.keys():
            gpu_id = self.param["gpu_id"]
        else:
            gpu_id = -1

        if "return_heatmap" in self.param.keys():
            self.return_heatmap = self.param["return_heatmap"]
        else:
            self.return_heatmap = False

        if torch.cuda.is_available() and gpu_id != -1:
            self.device = 'cuda:{}'.format(gpu_id)
        else:
            self.device = 'cpu'

        self.model = DRNSeg(2)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)
        self.model.eval()

        # Data preprocessing
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        self.input_data = self.uniform_input()

        for input_img in self.input_data:

            im_w = input_img.shape[1]
            im_h = input_img.shape[0]

            input_img = Image.fromarray(cv2.cvtColor(input_img,
                                                     cv2.COLOR_BGR2RGB))
            face = self.resize_shorter_side(input_img, 400)[0]
            face_tens = self.tf(face).to(self.device)

            # Warping field prediction
            with torch.no_grad():
                flow = self.model(face_tens.unsqueeze(0))[0].cpu().numpy()
                flow = np.transpose(flow, (1, 2, 0))
                h, w, _ = flow.shape

            # Undoing the warps
            modified = face.resize((w, h), Image.BICUBIC)
            modified_np = np.asarray(modified)
            reverse_np = self.warp(modified_np, flow)
            reverse_np = cv2.resize(reverse_np, (im_w, im_h),
                                    interpolation=cv2.INTER_CUBIC)

            if self.return_heatmap:
                flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
                traces_heatmap = self.get_heatmap_cv(modified_np, flow_magn, 7)
                traces_heatmap = cv2.resize(traces_heatmap, (im_w, im_h),
                                            interpolation=cv2.INTER_CUBIC)
                traces_heatmap = cv2.cvtColor(
                    traces_heatmap, cv2.COLOR_RGB2BGR)
                self.output_data.append(traces_heatmap)
            else:
                reverse_np = cv2.cvtColor(reverse_np, cv2.COLOR_RGB2BGR)
                self.output_data.append(reverse_np)

        self.uniform_output()

    def resize_shorter_side(self, img, min_length):
        """
        Resize the shorter side of img to min_length while
        preserving the aspect ratio.
        """
        ow, oh = img.size
        mult = 8
        if ow < oh:
            if ow == min_length and oh % mult == 0:
                return img, (ow, oh)
            w = min_length
            h = int(min_length * oh / ow)
        else:
            if oh == min_length and ow % mult == 0:
                return img, (ow, oh)
            h = min_length
            w = int(min_length * ow / oh)
        return img.resize((w, h), Image.BICUBIC), (w, h)

    def warp(self, im, flow, alpha=1, interp=cv2.INTER_CUBIC):
        height, width, _ = flow.shape
        cart = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
        pixel_map = (cart + alpha * flow).astype(np.float32)
        warped = cv2.remap(
            im,
            pixel_map[:, :, 0],
            pixel_map[:, :, 1],
            interp,
            borderMode=cv2.BORDER_REPLICATE)
        return warped

    def get_heatmap_cv(self, img, magn, max_flow_mag):
        min_flow_mag = .5
        cv_magn = np.clip(
            255 * (magn - min_flow_mag) / (max_flow_mag - min_flow_mag),
            a_min=0,
            a_max=255).astype(np.uint8)
        if img.dtype != np.uint8:
            img = (255 * img).astype(np.uint8)

        heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
        heatmap_img = heatmap_img[..., ::-1]

        h, w = magn.shape
        img_alpha = np.ones((h, w), dtype=np.double)[:, :, None]
        heatmap_alpha = np.clip(
            magn / max_flow_mag, a_min=0, a_max=1)[:, :, None]**.7
        heatmap_alpha[heatmap_alpha < .2]**.5
        pm_hm = heatmap_img * heatmap_alpha
        pm_img = img * img_alpha
        cv_out = pm_hm + pm_img * (1 - heatmap_alpha)
        cv_out = np.clip(cv_out, a_min=0, a_max=255).astype(np.uint8)

        return cv_out
