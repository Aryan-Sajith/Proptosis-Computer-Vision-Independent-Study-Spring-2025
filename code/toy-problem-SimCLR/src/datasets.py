import os, json, random
from PIL import Image
from torch.utils.data import Dataset

class UTKFaceDataset(Dataset):
    """Loads UTKFace; if mode='ssl', returns unlabeled images;
       if mode in {'train','val','test'} returns (img, label)."""
    def __init__(self, root_dir, splits_file, mode, transform=None, age_bins=(20,60)):
        """
        splits_file: JSON with keys 'train','val','test' listing filenames.
        mode: 'ssl' or one of the few-shot splits.
        age_bins: (young_max, old_min)
        """
        self.root = root_dir
        self.mode = mode
        self.transform = transform
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        if mode == 'ssl':
            # use *all* images except those in few-shot splits
            excluded = set(sum(splits.values(), []))
            self.fnames = [f for f in os.listdir(root_dir) if f not in excluded]
            self.labels = None
        else:
            self.fnames = splits[mode]
            # parse age from filename: "[age]_[gender]_[race]_[...].jpg"
            ages = [int(f.split('_')[0]) for f in self.fnames]
            y_max, o_min = age_bins
            self.labels = [0 if age < y_max else 1 for age in ages]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.mode == 'ssl':
            return img
        else:
            return img, self.labels[idx]

def make_fewshot_splits(root_dir, out_file, seed=42,
                        k_per_class=100, age_bins=(20,60),
                        val_per_class=10, test_per_class=10):
    """
    Sample k images per bin (<y_max, >o_min), then split into train/val/test.
    Saves JSON with lists of filenames.
    """
    random.seed(seed)
    all_files = os.listdir(root_dir)
    young, old = [], []
    for f in all_files:
        try:
            age = int(f.split('_')[0])
        except:
            continue
        if age < age_bins[0]:
            young.append(f)
        elif age > age_bins[1]:
            old.append(f)
    few_y = random.sample(young, k_per_class)
    few_o = random.sample(old, k_per_class)
    splits = {'train': [], 'val': [], 'test': []}
    def split_class(lst):
        random.shuffle(lst)
        return lst[val_per_class+test_per_class:], lst[:val_per_class], lst[val_per_class:val_per_class+test_per_class]
    ty, vy, ty_t = split_class(few_y)
    to, vo, to_t = split_class(few_o)
    splits['train'] = ty + to
    splits['val']   = vy + vo
    splits['test']  = ty_t + to_t
    with open(out_file, 'w') as f:
        json.dump(splits, f, indent=2)
