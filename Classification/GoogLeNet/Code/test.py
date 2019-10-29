import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
import json

from DataProcess.data_config import config
from DataProcess.data_loader import get_folders, MyDataSet, TestDataSet, get_img_dict
# from googlenet import googlenet


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    #=======================================
    #           1. Load dataset
    #=======================================
    test_datasets = TestDataSet(config.test_data, transforms=None)
    test_loader = DataLoader(dataset=test_datasets,
                             batch_size=config.batch_size, shuffle=False)
    print("Test numbers: {:d}".format(len(test_datasets)))

    #=======================================
    #   2. Define network and Load model
    #=======================================
    model = models.resnet101(num_classes=config.num_classes)
    model.load_state_dict(torch.load(config.model_path))
    print("load model success")
    model.to(device)

    #=======================================
    #         4. Test the dataset
    #=======================================
    model.eval()

    # result dict for json file
    results = get_img_dict(config.test_data)

    with torch.no_grad():
        for i, images in enumerate(test_loader, 0):

            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # put predicted classes to dict
            for j in range(predicted.size(0)):
                results[i * config.batch_size +
                        j]['disease_class'] = predicted[j].item()

    # Save to json file
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    with open(config.submit_path, 'w') as f:
        json.dump(results, f)
    print("save json file to submit success")


if __name__ == '__main__':

    main()
