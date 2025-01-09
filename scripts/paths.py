import subprocess
import os
import yaml
WORKING_DIR = subprocess.check_output("find ~ -name rl_tb3 | grep rl_tb3",
                            shell=True,
                            executable="/bin/bash").decode().rstrip('\n')

CHECKPOINT_DIR = f'{WORKING_DIR}/checkpoints'
MISC_DIR = f'{WORKING_DIR}/misc'

dirs = [CHECKPOINT_DIR, MISC_DIR]

configs = {
    'status': 'not_started',
    'max_reward': -1000.0,
    'epochs': 0,
    'checkpoint_path': '',
    'configs': None
}

for dir in dirs:
    if os.path.exists(dir):
        pass
    else:
        os.mkdir(dir)
        print(f'Created {dir} folder...')
        if dir == MISC_DIR:
            with open(f'{MISC_DIR}/misc.yaml', 'w') as file:
                yaml.safe_dump(configs, file)



    