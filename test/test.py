from yaml import safe_load

def make_net(configs):
    if configs['network_config']['net_type'] == 'separate':
        configs['network_config']['actor'] = 'actor'
        configs['network_config']['critic'] = 'critic'
    else:
        configs[configs['network_config']['actor']] = 'shared'


file = open('config.yaml')


data = safe_load(file)

file.close()

print(data['network_config'])

make_net(data)

print(data['network_config'])






