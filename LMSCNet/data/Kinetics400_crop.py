from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torchvision
from random import randint
import os
import csv
import sys

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class Kinetics400(VisionDataset):

    def __init__(self, root, file_root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0):
        super(Kinetics400, self).__init__(root)
        
        data_root = os.path.join(root, 'datas')
        self.classes, class_to_idx = find_classes(data_root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        #csv_root = '/home/bryan/Desktop/test_data'
        
        with open(os.path.join(root, file_root), newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            self.video_list = [os.path.join(data_root, row['label'], '%s_%s_%s.mp4' 
                                  % (row['youtube_id'],
                                 '%06d' % int(row['time_start']),
                                 '%06d' % int(row['time_end']))) for row in rows]

        with open(os.path.join(root, file_root), newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            self.label_list = [class_to_idx[row['label']] for row in rows]  

        '''self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )'''
        self.transform = transform
        #self.count = 0

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return len(self.video_list)
        # return 100

    def __getitem__(self, idx):
        video = torchvision.io.read_video(self.video_list[idx])[0]
        label = self.label_list[idx]
        
        while video.shape[0]<17:
            idx +=1
            #self.count +=1
            #sys.stdout.write("\rprogress:\t{}/{}".format(self.count,len(self.video_list))+"\n")
            #sys.stdout.flush()
            video = torchvision.io.read_video(self.video_list[idx])[0]
            label = self.label_list[idx]

            #padding = torch.randint(255,(16,360,480,3)).byte()
            #label = self.label_list[idx]
        
        try:
            if self.transform is not None:
                video = video.permute(3,0,1,2)
                video = self.transform(video)
                video = video.permute(1,2,3,0)
                rand_t = randint(0, video.shape[0]-16)
                video = video[rand_t:rand_t+16]
        except ValueError:
            print(idx)

        data = {}
        data['3D_OCCUPANCY'] = video
        data['3D_LABEL'] = label

        return data, idx