from torchvision import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = T.Compose([
            T.RandomHorizontalFlip(),  # 随机水平反转
            T.RandomRotation(30),  # 随机旋转
            T.RandomResizedCrop(cfg.input_shape),  # 随机裁剪并调整到输入大小
            # T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.input_shape),
            # T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform