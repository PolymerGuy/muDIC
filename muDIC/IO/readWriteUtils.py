import dill


def save(dic, name):
    try:
        with open(name + '.p', 'wb') as myfile:
            dill.dump(dic, myfile)
    except:
        raise IOError('Could not save to file')


def load(name):
    try:
        with open(name + '.p', 'rb') as myfile:
            return dill.load(myfile)
    except:
        raise TypeError("Invalid inputs")
