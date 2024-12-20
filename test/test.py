from yaml import load


file = open('config.yaml')


data = load(file)


print(data)