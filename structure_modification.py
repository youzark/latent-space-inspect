from model import CNN

def rearrage_feature_dimension(
        model:CNN,
        layer_idx:int,
        new_indices:list[int]
    ):
    model.layers[layer_idx-1].rearrange_output(new_indices)
    model.layers[layer_idx].rearrange_input(new_indices)


def truncate_feature_dimension(
    model:CNN,
    layer_idx:int,
    num_trunc:int=1
    ):
    model.layers[layer_idx-1].truncate_output(num_trunc)
    model.layers[layer_idx].truncate_input(num_trunc)
