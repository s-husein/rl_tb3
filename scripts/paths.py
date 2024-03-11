from os import environ
home_folder = environ.get("HOME")
PARENTFOLDER = f'{home_folder}/fyp/src/rl_tb3'

PLOTFOLDER = f'{PARENTFOLDER}/plots'
MODELFOLDER = f'{PARENTFOLDER}/models'
CHECKPOINTFOLDER = f'{PARENTFOLDER}/checkpoints'
STATUSFILE= f'{PARENTFOLDER}/status.txt'
REWARDFOLDER = f'{PARENTFOLDER}/rewards'
CONFIGFOLDER = f'{PARENTFOLDER}/configs'