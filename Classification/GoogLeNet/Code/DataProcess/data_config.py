class DefaultConfigs(object):
    #=======================================
    #        1. String parameters
    #=======================================
    train_data = "data/traffic-sign/train/"
    valid_data = "data/traffic-sign/valid/"
    test_data = "data/traffic-sign/test/"
    # model_path = "models/googlenet.pth"
    model_path = "checkpoints/googlenet_model-98.833.pth"
    checkpoint = "checkpoints/"
    best_models = checkpoint + "googlenet_model"
    submit = "submit/"
    submit_path = submit + "googlenet.json"
    gpus = "0"
    pretrained = False

    #=======================================
    #        2. Numeric parameters
    #=======================================
    epochs = 2
    batch_size = 16
    img_weight = 500
    img_height = 500
    num_classes = 10
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4


config = DefaultConfigs()
