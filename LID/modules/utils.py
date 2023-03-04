
def str2bool(s):
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise NotImplementedError