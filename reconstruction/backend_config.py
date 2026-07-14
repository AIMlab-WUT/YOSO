from backends import NumpyBackend, TFBackend

DEFAULT_BACKEND = "tf"


def get_backend(name=DEFAULT_BACKEND):
    if name == "numpy":
        print("Set numpy backend")
        return NumpyBackend()
    elif name == "tf":
        print("Set tensorflow backend")
        return TFBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")