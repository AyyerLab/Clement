import numpy as np

def my_print(*args):
    print(args)
    do_print(*args)

def do_print(*args):
    print(args)

strings = ['A', 'B', 'C']

#my_print(*strings)


class A():
    def __init__(self):
        self.print = 5

    def printer(self):
        self.print = self.parse

    def parse(self):
        print('hello')

