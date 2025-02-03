from .model import msnet26, msnet50, msnet101


def get_model(name, **kwargs):

    if name == "ms26":
        return msnet50()
    elif name == "ms50":
        return msnet101()
    elif name == "ms101":
        return msnet101()

    else:
        raise ValueError()
