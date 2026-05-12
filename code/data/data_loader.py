import os, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomCrop
from torchvision.transforms.functional import crop, rotate

class TrainDataset(Dataset):
    def __init__(self, snow_path, clear_path):
        super(TrainDataset, self).__init__()
        self.snow_path = snow_path
        self.clear_path = clear_path
        self.snow_images_list = os.listdir(self.snow_path)
        self.clear_images_list = os.listdir(self.clear_path)
        self.totensor = ToTensor()
    def __getitem__(self, index):
        snow_image_name = self.snow_images_list[index]
        clear_image_name = self.clear_images_list[index]
        snow_image_path = os.path.join(self.snow_path, snow_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        snow_image = Image.open(snow_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        crop_params = RandomCrop.get_params(snow_image, [196,196])
        rotate_params = random.randint(0,3) * 90

        snow = crop(snow_image, *crop_params)
        clear = crop(clear_image, *crop_params)

        snow = rotate(snow, rotate_params)
        clear = rotate(clear, rotate_params)

        snow = self.totensor(snow)
        clear = self.totensor(clear)
        return snow, clear
    def __len__(self):
        return len(self.snow_images_list)

class TestDataset(Dataset):
    def __init__(self, snow_path, clear_path, max_images=None, crop_size=(196, 196)):
        super(TestDataset, self).__init__()
        self.snow_path = snow_path
        self.clear_path = clear_path
        self.snow_images_list = os.listdir(self.snow_path)
        self.clear_images_list = os.listdir(self.clear_path)
        self.totensor = ToTensor()
        self.snow_images_list.sort()
        self.clear_images_list.sort()
        if max_images is not None:
            self.snow_images_list = self.snow_images_list[:max_images]
            self.clear_images_list = self.clear_images_list[:max_images]
        self.crop_size = crop_size
    def __getitem__(self, index):
        snow_image_name = self.snow_images_list[index]
        clear_image_name = self.clear_images_list[index]
        snow_image_path = os.path.join(self.snow_path, snow_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        snow_image = Image.open(snow_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        if self.crop_size is not None:
            crop_params = RandomCrop.get_params(snow_image, self.crop_size)
            snow_image = crop(snow_image, *crop_params)
            clear_image = crop(clear_image, *crop_params)

        snow = self.totensor(snow_image)
        clear = self.totensor(clear_image)
        return snow, clear, snow_image_name
    def __len__(self):
        return len(self.snow_images_list)

class ValDataset(Dataset):
    def __init__(self, snow_path, clear_path):
        super(ValDataset, self).__init__()
        self.snow_path = snow_path
        self.clear_path = clear_path
        self.snow_images_list = os.listdir(self.snow_path)
        self.clear_images_list = os.listdir(self.clear_path)
        self.totensor = ToTensor()
        self.snow_images_list.sort()
        self.clear_images_list.sort()
    def __getitem__(self, index):
        snow_image_name = self.snow_images_list[index]
        clear_image_name = self.clear_images_list[index]
        snow_image_path = os.path.join(self.snow_path, snow_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        snow_image = Image.open(snow_image_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')

        snow = self.totensor(snow_image)
        clear = self.totensor(clear_image)
        return {'snow': snow, 'clear': clear, 'filename': clear_image_name}
    def __len__(self):
        return len(self.snow_images_list)

class ValRealDataset(Dataset):
    def __init__(self, snow_path):
        super(ValRealDataset, self).__init__()
        self.snow_path = snow_path
        self.snow_images_list = os.listdir(self.snow_path)
        self.totensor = ToTensor()
        self.snow_images_list.sort()
    def __getitem__(self, index):
        snow_image_name = self.snow_images_list[index]
        snow_image_path = os.path.join(self.snow_path, snow_image_name)
        snow_image = Image.open(snow_image_path).convert('RGB')

        snow = self.totensor(snow_image)
        return {'snow': snow, 'filename': snow_image_name}
    def __len__(self):
        return len(self.snow_images_list)






