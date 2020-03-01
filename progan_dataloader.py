import torch
import torchvision
from PIL import Image, ImageOps
import os, random


class CelebDataset(torch.utils.data.Dataset):
    """
    A datset which loads images from a flat directory and scales them
    to a size specified when the dataloader is created. We create a new 
    loader every time we change resolutions in training.
    """
    def __init__(self, source_directory, resize_directory, resolution):
        # This will resize the image to the target resolution and normalize to the
        # range [-1, 1] (the Normalize transform will subtract 0.5 from each color
        # channel, then divide by 0.5).
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.directory = os.path.join(resize_directory, str(resolution))

        # Resize the images to each size - an example of how you might do preprocessing
        # and storage on-disk. We still do normalization on the fly, but you could
        # avoid that by storing raw tensors instead of images. The if assumes that if
        # the directory is empty or nonexistent we need to do the resizing, but does
        # not handle partially-full directories or invalid files.
        if not os.path.isdir(self.directory) or len(os.listdir(self.directory)) == 0:
            os.makedirs(self.directory, exist_ok=True)
            originals = [file
                         for file in os.listdir(source_directory)
                         if os.path.isfile(os.path.join(source_directory, file))]
            # We assume all images are 1024 to begin with; handling the general case
            # would really complicate this a lot more than necessary.
            downsample = torch.nn.AvgPool2d(1024 // resolution)
            toImage = torchvision.transforms.ToPILImage()
            for sample in originals:
                image = Image.open(os.path.join(source_directory, sample))
                image = self.toTensor(image)
                image = downsample(image)
                image = toImage(image)
                png_ext = ".".join(sample.split(".")[:-1]) + ".png"
                image.save(os.path.join(self.directory, png_ext), "PNG")

        # Assume that every regular file in the directory is a sample
        self.samples = [
            file
            for file in os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, file))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.directory, self.samples[index]))
        if random.random() > 0.5:
            image = ImageOps.mirror(image)
        return self.transform(image)
