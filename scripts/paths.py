import subprocess
import os
import yaml
WORKING_DIR = subprocess.check_output("find ~ -name rl_tb3 | grep src/rl_tb3",
                            shell=True,
                            executable="/bin/bash").decode().rstrip('\n')

CHECKPOINT_DIR = f'{WORKING_DIR}/checkpoints'
MISC_DIR = f'{WORKING_DIR}/misc'

dirs = [CHECKPOINT_DIR, MISC_DIR]

with open(f'{WORKING_DIR}/config.yaml') as file:
    params = yaml.safe_load(file)

configs = {
    'status': 'not_started',
    'max_reward': params['algo_params']['max_reward'],
    'epochs': 0
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



    