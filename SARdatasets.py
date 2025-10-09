import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import numpy as np
import random
from scipy.ndimage import convolve


class SARImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)

        edges = cv2.Canny(image.astype(np.uint8), 200, 300)

        corners = cv2.cornerHarris(image, 5, 3, 0.04)
        corners = corners * 255

        multi_channel_image = np.dstack((image, edges, corners))
        multi_channel_image = multi_channel_image.astype(np.uint8)
        multi_channel_image = Image.fromarray(multi_channel_image)

        if self.transform is not None:
            multi_channel_image = self.transform(multi_channel_image)

        return multi_channel_image, target


class build_coed_SARImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]

        image_3ch = Image.open(path).convert('RGB')
        image = Image.open(path).convert('L')
        image_np = np.array(image)

        edges = cv2.Canny(image_np, 200, 300)

        corners = cv2.cornerHarris(image_np, 5, 3, 0.04)
        corners = corners * 255

        multi_channel_image = np.dstack((image_np, edges, corners))
        multi_channel_image = multi_channel_image.astype(np.uint8)
        multi_channel_image = Image.fromarray(multi_channel_image)

        if self.transform is not None:
            multi_channel_image = self.transform(multi_channel_image)
            image_3ch = self.transform(image_3ch)

        target = multi_channel_image

        return image_3ch, target


class Multi_task_SARImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def add_gamma_noise(self, image_np, looks):
        """
        向图像添加伽马分布的相干斑噪声
        :param image_np: 原始图像的numpy数组
        :param looks: SAR图像的等效视数(ENL,越大噪声越小)
        :return: 加噪后的图像
        """
        image_np = image_np.astype(np.float32)

        image_np = image_np / np.max(image_np)

        gamma_noise = np.random.gamma(shape=looks, scale=1.0 / looks, size=image_np.shape)

        noisy_image = image_np * gamma_noise

        noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

        return noisy_image

    def add_gaussian_noise(self, image_np, snr_db):
        """
        向图像添加高斯白噪声
        :param image_np: 原始图像的numpy数组
        :param snr_db: 期望的信噪比（以分贝为单位）
        :return: 加噪后的图像
        """
        signal_power = np.mean(image_np ** 2)

        snr = 10 ** (snr_db / 10)

        noise_power = signal_power / snr

        noise_sigma = np.sqrt(noise_power)

        current_state = torch.random.get_rng_state()
        current_cuda_state = torch.cuda.get_rng_state()

        torch.manual_seed(np.random.randint(0, 2 ** 31 - 1))
        torch.cuda.manual_seed_all(np.random.randint(0, 2 ** 31 - 1))

        noise = np.random.normal(0, noise_sigma, image_np.shape)

        torch.random.set_rng_state(current_state)
        torch.cuda.set_rng_state(current_cuda_state)

        noisy_image = image_np + noise

        return noisy_image.astype(np.uint8)

    def log_transform(self, image_np):
        image_np = image_np.astype(np.float32)

        c = 20.0  
        transformed_image = c * np.log1p(image_np)  # torch.log1p计算log(1 + x)

        return transformed_image

    def __getitem__(self, index):
        path, target = self.samples[index]

        image_3ch = Image.open(path).convert('RGB')
        image_3ch_np = np.array(image_3ch)
        
        image = Image.open(path).convert('L')
        image_np = np.array(image)

        edges = cv2.Canny(image_np, 200, 300)

        corners = cv2.cornerHarris(image_np, 5, 3, 0.04)
        corners = corners * 255

        first_channel = image_3ch_np[:, :, 0]
        noisy_first_channel = self.add_gamma_noise(first_channel, 30)
        image_3ch_np[:, :, 0] = noisy_first_channel
        image_3ch = Image.fromarray(image_3ch_np)

        multi_channel_image = np.dstack((image_np, edges, corners))
        multi_channel_image = multi_channel_image.astype(np.uint8)
        multi_channel_image = Image.fromarray(multi_channel_image)

        if self.transform is not None:
            multi_channel_image = self.transform(multi_channel_image)
            image_3ch = self.transform(image_3ch)

        target = multi_channel_image

        return image_3ch, target
    

class Multi_task_angel_SARImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def add_gaussian_noise(self, image_np, snr_db):

        signal_power = np.mean(image_np ** 2)

        snr = 10 ** (snr_db / 10)

        noise_power = signal_power / snr

        noise_sigma = np.sqrt(noise_power)

        noise = np.random.normal(0, noise_sigma, image_np.shape)

        noisy_image = image_np + noise

        return noisy_image.astype(np.uint8)

    def log_transform(self, image_np):

        image_np = image_np.astype(np.float32)


        c = 20.0 
        transformed_image = c * np.log1p(image_np)

        return transformed_image

    def __getitem__(self, index):
        path, target = self.samples[index]

        image_3ch = Image.open(path).convert('RGB')
        image_3ch_np = np.array(image_3ch)
        
        image = Image.open(path).convert('L')
        image_np = np.array(image)

        edges = cv2.Canny(image_np, 200, 300)

        corners = cv2.cornerHarris(image_np, 5, 3, 0.04)
        corners = corners * 255

        kernel_size = 50  
        kernel = np.ones((kernel_size, kernel_size))
        density = convolve(corners, kernel, mode='constant', cval=0.0)

        max_density_index = np.unravel_index(np.argmax(density), density.shape)
        center_y, center_x = max_density_index

        half_size = kernel_size // 2
        start_y = max(center_y - half_size, 0)
        end_y = min(center_y + half_size, corners.shape[0])
        start_x = max(center_x - half_size, 0)
        end_x = min(center_x + half_size, corners.shape[1])

        region = image_np[start_y:end_y, start_x:end_x]

        angle = random.choice([0, 90, 180, 270])
        M = cv2.getRotationMatrix2D((region.shape[1] // 2, region.shape[0] // 2), angle, 1)
        rotated_region = cv2.warpAffine(region, M, (region.shape[1], region.shape[0]))

        rotated_image = image_np.copy()
        rotated_image[start_y:end_y, start_x:end_x] = rotated_region

        image_4ch_np = np.insert(image_3ch_np, 1, rotated_image, axis=2)

        first_channel = image_3ch_np[:, :, 0]
        first_channel = self.log_transform(first_channel)
        noisy_first_channel = self.add_gaussian_noise(first_channel, 30)
        image_4ch_np[:, :, 0] = noisy_first_channel
        image_4ch = Image.fromarray(image_3ch_np)

        multi_channel_image = np.dstack((image_np, image_np, edges, corners))
        multi_channel_image = multi_channel_image.astype(np.uint8)
        multi_channel_image = Image.fromarray(multi_channel_image)

        if self.transform is not None:
            multi_channel_image = self.transform(multi_channel_image)
            image_4ch = self.transform(image_4ch)

        target = image_4ch

        return multi_channel_image, target