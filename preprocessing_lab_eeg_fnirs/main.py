import numpy as np
import yaml
import data_loader



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data = data_loader.load(config)

    print(config['defaults'])



    pass