import numpy as np
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))
targets = np.array([1, 1, 1, 4, 4, 4, 0, 2, 3, 1])
orders = [4, 3, 2, 1, 0]
if __name__ == '__main__':
    print(_map_new_class_index(targets, orders))