from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
import csv

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class Kinetics400(VisionDataset):

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0):
        super(Kinetics400, self).__init__(root)

        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        #video_list = [x[0] for x in self.samples]
        with open('/media/public_dataset/Kinetics400/kinetics-400_train.csv', newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            video_list = [os.path.join(root, row['label'], '%s_%s_%s.mp4' 
                                  % (row['youtube_id'],
                                 '%06d' % int(row['time_start']),
                                 '%06d' % int(row['time_end']))) for row in rows]

            self.label_list = [class_to_idx[row['label']] for row in rows]
        
        self.video_clips = VideoClips(
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
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = video.permute(3,0,1,2)
            video = self.transform(video)
            video = video.permute(1,2,3,0)

        data={}
        data['3D_OCCUPANCY'] = video
        data['3D_LABEL'] = label

        return data, idx