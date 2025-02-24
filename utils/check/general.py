def each_element_in_list(list_to_check, containing_list, return_unexpected_elements: bool = False):
    list_unexpected = []
    for element in list_to_check:
        if element not in containing_list:
            if return_unexpected_elements:
                list_unexpected.append(element)
            else:
                return False
    if return_unexpected_elements:
        return bool(not len(list_unexpected)), list_unexpected
    else:
        return True


class Checker:
    def __init__(self, name_target):
        self.name_target = name_target

    def __call__(self,x):
        check_name(x, self.name_target)


def check_name(x, name_target):
    try:
        name = x.__name__
    except:
        return False
    return name_target in name
