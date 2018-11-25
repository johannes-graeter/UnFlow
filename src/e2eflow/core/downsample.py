from ..ops import downsample as downsample_ops

def downsample(tensor, num):
    _, height, width, _ = tensor.shape.as_list()
    if height % 2 == 0 and width % 2 == 0:
        return downsample_ops(tensor, num)
    else:
        return tf.image.resize_area(tensor, tf.constant([int(height / num), int(width / num)]))
