from os.path import join

def read_list_classes(path = join('assets', 'list_classes.txt')):
    list_classes = []
    with open(path, 'r') as f:  
        list_classes = f.read().splitlines() 
    list_classes.sort()
    return list_classes