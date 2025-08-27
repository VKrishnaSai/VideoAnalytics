# I am excellent in all regards coz I am the great Unni Krishnan
# Anyone who dares question my superiority will be challenged by the legendary Sai Krishnan.
# All you puny mortals shall perish and we shall reign glory across the nine realms.


import pathlib
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from transformers import VivitImageProcessor, VivitForVideoClassification
from transformers import TrainingArguments, Trainer
import evaluate
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from accelerate import Accelerator
from accelerate.data_loader import IterableDatasetShard

class UCF101Dataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        video = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        
        # Get video duration and sample a clip
        duration = video.duration
        start_sec = 0
        end_sec = duration
        
        # Read video data
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = video_data['video']
        
        if self.transform:
            transformed = self.transform({"video": video_data})
            video_data = transformed["video"]
        
        return {"video": video_data, "label": label}

def load_ucf101_splits(split_files_path, ucf_path):
    """Load official UCF101 splits and create train/val/test datasets"""
    
    # Read class mapping
    class_mapping = {}
    with open(os.path.join(split_files_path, 'classInd.txt'), 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split()
            class_mapping[class_name] = int(class_id) - 1  # 0-based indexing
    
    # Read train split (using split 01)
    train_videos = []
    train_labels = []
    with open(os.path.join(split_files_path, 'trainlist01.txt'), 'r') as f:
        for line in f:
            video_path, class_id = line.strip().split()
            full_path = os.path.join(ucf_path, video_path)
            if (os.path.exists(full_path)):
                train_videos.append(full_path)
                class_name = video_path.split('/')[0]
                train_labels.append(class_mapping[class_name])
    
    # Read test split (using split 01)
    test_videos = []
    test_labels = []
    with open(os.path.join(split_files_path, 'testlist01.txt'), 'r') as f:
        for line in f:
            video_path = line.strip()
            full_path = os.path.join(ucf_path, video_path)
            if os.path.exists(full_path):
                test_videos.append(full_path)
                class_name = video_path.split('/')[0]
                test_labels.append(class_mapping[class_name])
    
    # Create validation split (20% of train data, stratified)
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        train_videos,
        train_labels,
        test_size=0.2,
        stratify=train_labels,
        random_state=42
    )
    
    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels), class_mapping

def preprocess_dataset(dataset_root_path, split_files_path, image_processor, num_frames_to_sample):
      mean = image_processor.image_mean
      std = image_processor.image_std
      if "shortest_edge" in image_processor.size:
            height = width = image_processor.size["shortest_edge"]
      else:
            height = image_processor.size["height"]
            width = image_processor.size["width"]
      resize_to = (height, width)

      # Load splits
      (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels), class_mapping = load_ucf101_splits(
          split_files_path, dataset_root_path
      )

      # Training dataset transformations
      train_transform = Compose(
                  [
                  ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                              [
                              UniformTemporalSubsample(num_frames_to_sample),
                              Lambda(lambda x: x / 255.0),
                              Normalize(mean, std),
                              RandomShortSideScale(min_size=256, max_size=320),
                              RandomCrop(resize_to),
                              RandomHorizontalFlip(p=0.5),
                              ]
                              ),
                        ),
                  ]
                  )

      # Training dataset
      train_dataset = UCF101Dataset(
                  video_paths=train_videos,
                  labels=train_labels,
                  transform=train_transform,
                  num_frames=num_frames_to_sample,
                  )

      # Validation and evaluation datasets' transformations
      val_transform = Compose(
            [
                  ApplyTransformToKey(
                  key="video",
                  transform=Compose(
                        [
                              UniformTemporalSubsample(num_frames_to_sample),
                              Lambda(lambda x: x / 255.0),
                              Normalize(mean, std),
                              Resize(resize_to),
                              ]
                        ),
                  ),
                  ]
            )

      # Validation and evaluation datasets
      val_dataset = UCF101Dataset(
                  video_paths=val_videos,
                  labels=val_labels,
                  transform=val_transform,
                  num_frames=num_frames_to_sample,
                  )

      test_dataset = UCF101Dataset(
                  video_paths=test_videos,
                  labels=test_labels,
                  transform=val_transform,
                  num_frames=num_frames_to_sample,
                  )

      return train_dataset, val_dataset, test_dataset

accelerator = Accelerator()
print("Process ID: %d of %d" % (accelerator.process_index, accelerator.num_processes))
print("Available GPU devices: %d" % torch.cuda.device_count())

dataset_root_path = "../UCF-101"
split_files_path = "../ucfTrainTestlist"
model_ckpt = "google/vivit-b-16x2-kinetics400"
batch_size = 4

image_processor = VivitImageProcessor.from_pretrained(model_ckpt)

# Get the initial splits and class mapping
(train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels), class_mapping = load_ucf101_splits(
    split_files_path, dataset_root_path
)

print(f"Train videos: {len(train_videos)}")
print(f"Validation videos: {len(val_videos)}")
print(f"Test videos: {len(test_videos)}")
print(f"Total videos: {len(train_videos) + len(val_videos) + len(test_videos)}")

# Create label mappings
label2id = {label: idx for label, idx in class_mapping.items()}
id2label = {idx: label for label, idx in class_mapping.items()}

print(f"Number of classes: {len(label2id)}")
print("First few classes:", list(label2id.keys())[:5])

train_dataset, val_dataset, test_dataset = (
    preprocess_dataset(dataset_root_path, split_files_path, image_processor, model.config.num_frames)
    )

# Training setup

model_name = model_ckpt.split("/")[-1]
# new_model_name = ("%s-finetuned-ucf101-subset-%s-n-%s-g-%d-b" %
#                   (model_name, os.getenv("SLURM_NNODES"), os.getenv("SLURM_GPUS_PER_NODE"), batch_size))
new_model_name = ('%s-finetuned-ucf101-subset-%s-n-%s-g-%d-b' % ("vivit-b-16x2-ucf101", 1, 1, batch_size))
args = TrainingArguments(
      new_model_name,
      remove_unused_columns=False,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      save_on_each_node=False,
      learning_rate=5e-5,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      warmup_ratio=0.1,
      logging_steps=10,
      load_best_model_at_end=True,
      metric_for_best_model="accuracy",
      push_to_hub=False,
      dataloader_num_workers=15, # Set it to 1 for single preprocess worker
      dataloader_prefetch_factor=64,
      max_steps=(train_dataset.num_videos // batch_size)*2,
)

# Next, we need to define a function for how to compute the metrics from the predictions,
# which will just use the metric we'll load now. The only preprocessing we have to do
# is to take the argmax of our predicted logits:
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
      """Computes accuracy on a batch of predictions."""
      predictions = np.argmax(eval_pred.predictions, axis=1)
      return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
      """The collation function to be used by `Trainer` to prepare data batches."""
      # permute to (num_frames, num_channels, height, width)
      pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
      )
      labels = torch.tensor([example["label"] for example in examples])
      return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
      model,
      args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=image_processor,
      compute_metrics=compute_metrics,
      data_collator=collate_fn,
)

train_results = trainer.train()

trainer.save_model()
test_results = trainer.evaluate(test_dataset)
trainer.log_metrics("test", test_results)
trainer.save_metrics("test", test_results)
trainer.save_state()