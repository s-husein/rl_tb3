import csv


# def x(**kwargs):

#     return kwargs

# print(x(a=1, b=2, c=3, d=4))



field_name = ['Rewards']

# with open('ex.csv', 'w') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=field_name)

#     writer.writeheader()

#     writer.writerow(['aa'])

data = []


with open('ex.csv') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        data.append(row['Rewards'])


print(data)

data = data[:5]

data_dict = {}


# with open('ex.csv', 'w') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=field_name)
#     writer.writeheader()
#     for ele in data:
#         data_dict = {}
#         for i, key in enumerate(field_name):
#             data_dict[key] = data


    # reader = csv.DictReader(csv_file)
    # rows = csv.reader(csv_file)
    # for row in reader:
    #     print(row)
    # for row in rows:
    #     print(row)







