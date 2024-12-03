from pathlib import Path

from data_loader.datasets.coco2017 import COCO2017
from config.config import Config


class Main:
    config_file_path = Path.cwd() / 'src' / 'config' / 'configs.yaml'

    def __init__(self):
        pass

    def bootstrap(self):
        config = Config()
        config.set_configs(self.config_file_path)

        dataset_configs = config.get_config('datasets.coco2017')
        model_configs = config.get_config('models.uvit')
        general_configs = config.get_config('general')
        data_loader = COCO2017(
            data_dir=dataset_configs['path'],
            batch_size=general_configs['batch_size'],
            image_size=model_configs['image_size']
        )

        data_loader.set_datasets()

        train_dataset = data_loader.get_train_dataset()
        val_dataset = data_loader.get_validation_data()


if __name__ == '__main__':
    Main().bootstrap()
