# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import  TubeMaskingGenerator

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    frame_id_list = np.arange(0, 32, 2) + 60  # Frame selection
    video_data = vr.get_batch(frame_id_list).asnumpy()
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

    # Transform video frames
    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None)) 
    img = img.view((args.num_frames, 3) + img.size()[-2:]).transpose(0, 1)
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img.unsqueeze(0).to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.unsqueeze(0).to(device, non_blocking=True).flatten(1).to(torch.bool)

        # 🔥 **Classification output**
        classification_logits = model(img, bool_masked_pos)
        class_probs = torch.nn.functional.softmax(classification_logits, dim=-1)
        top5_prob, top5_classes = torch.topk(class_probs, 5, dim=-1)

        # 🔥 **Map class indices to UCF-101 labels**
        ucf101_labels = load_ucf101_labels("../../ucfTrainTestlist/classInd.txt") 
        print('DEBUGGING----------------------------') 
        print(top5_classes)
        top5_class_names = [[ucf101_labels[int(idx)] for idx in frame_top5] for frame_top5 in top5_classes.squeeze(0).tolist()]
        print("\n🎯 **Top-5 Predicted Classes:**")
        print(top5_class_names)

def load_ucf101_labels(filepath):
    """Load UCF-101 class labels from classInd.txt (1-based index)."""
    labels = {}
    with open(filepath, "r") as f:
        for line in f:
            index, name = line.strip().split()
            labels[int(index) - 1] = name  # Convert to 0-based index
    return labels


if __name__ == '__main__':
    opts = get_args()
    main(opts)
    # ucf101_labels = load_ucf101_labels("../../ucfTrainTestlist/classInd.txt")  
    # print(ucf101_labels[0])
