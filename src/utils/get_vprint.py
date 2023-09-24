def get_vprint(verbose: bool = True) -> callable:
    """
    Function to get a print function that prints only if verbose is True

    :param verbose: Boolean argument to print or not
    :return: Callable function to print
    :author: Ricardo Espantaleón Pérez
    """
    vprint = print if verbose else lambda *a, **k: None

    return vprint
