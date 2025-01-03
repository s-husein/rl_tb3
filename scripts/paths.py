import subprocess
WORKING_DIR = subprocess.check_output("find ~ -name rl_tb3 | grep src/rl_tb3",
                            shell=True,
                            executable="/bin/bash").decode().rstrip('\n')

PLOTFOLDER = f'{WORKING_DIR}/plots'
MODELFOLDER = f'{WORKING_DIR}/models'
CHECKPOINTFOLDER = f'{WORKING_DIR}/checkpoints'
STATUSFILE= f'{WORKING_DIR}/status.txt'
REWARDFOLDER = f'{WORKING_DIR}/rewards'
CONFIGFOLDER = f'{WORKING_DIR}/configs'
RESULTFOLDER = f'{WORKING_DIR}/result'