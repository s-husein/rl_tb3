def fun(a, b, c, d, e):
    return a, b, c, d, e


x = {'c': 3, 'd': 4, 'b': 2, 'a': 1}

print(fun(**x, e=10))





