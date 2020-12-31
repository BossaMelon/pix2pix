import getpass
import os
from datetime import datetime
from pathlib import Path

current_username = getpass.getuser()
project_root = Path(__file__).parent.parent.resolve()
project_name = project_root.name

if current_username == 'wyh':
    project_resource_root = project_root / 'resources'
    data_path = project_resource_root / 'data'
    pretrained_model_path = project_resource_root / 'pretrained_models'
    model_checkpoint_path = project_resource_root / 'model_checkpoints'

elif current_username == 'wangyueh':
    tnt_resource_root = Path('/home/wangyueh/tmp/project_resources')
    project_resource_root = tnt_resource_root / project_name
    ssd_root = Path('/phys/ssd/wangyueh')
    data_path = ssd_root / project_name / 'data'
    pretrained_model_path = ssd_root / project_name / 'pretrained_models'
    model_checkpoint_path = project_resource_root / 'model_checkpoints'

else:
    raise Exception('no valid data path')


def _get_result_path():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H%M%S")
    result_root = project_resource_root / 'results' / dt_string
    return result_root


result_root_path = _get_result_path()
visualization_path = result_root_path / 'visualization'


def create_folder():
    folders = [project_resource_root, data_path, visualization_path, data_path, pretrained_model_path, model_checkpoint_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


create_folder()

if __name__ == '__main__':
    print(Path(__file__).parent.parent.resolve())
