class cfg:
    data_root = r"deeplearning\UNet\data"
    in_channels = 3
    num_classes = 1       # >1 时用 CE/softmax
    base_channels = 64
    batch_size = 4
    lr = 1e-3
    epochs = 50
    num_workers = 4
    loss_name = "bce+dice"
    lr_step = 20
    lr_gamma = 0.5
