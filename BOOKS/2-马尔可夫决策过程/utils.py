
def str_key(*args):
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return '_'.join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value

def set_prob(P, s, a, s1, p = 1.0):
    set_dict(P, p, s, a, s1)

def get_prob(P, s, a, s1):
    P.get(str_key(s, a, s1), 0)

