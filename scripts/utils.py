from paths import PLOTFILE, CHECKPOINTFILE, STATUSFILE
import pickle as pik


def read_file(path, byte=False):
    mode = 'r'
    if byte:
        mode = 'rb'
    file = open(path, mode)
    file.seek(0)
    info = file.readline()
    file.close()
    return info

def write_file(self, path, content):
        mode = 'w'
        if path == PLOTFILE:
            mode = '+a'
        file = open(path, mode=mode)
        file.write(content)
        file.close()
    

def create_file(num):
    path = f'{CHECKPOINTFILE}/checkpoint_{num}.pth'
    file = open(path, 'w')
    file.close()
    return path

def save_status_dict(data_dict):
    file = open(STATUSFILE, 'wb')
    pik.dump(data_dict, file)
    file.close()

def load_status_dict():
    file = open(STATUSFILE, 'rb')
    data = pik.load(file=file)
    file.close()
    return data

def check_status_file(epochs):
    if read_file(STATUSFILE, byte=True) == b'':
        file = open(STATUSFILE, 'wb')
        save_status_dict({'path': '', 'num':epochs})
        file.close()
    else: pass