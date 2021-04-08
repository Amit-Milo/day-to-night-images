import matplotlib.pyplot as plt

from image_utils import tensor_to_numpy



def plot_result(result):
    fig, ax1 = plt.subplots(1, figsize=(20, 10))
    plt.axis('off')
    # content and style ims side-by-side
    ax1.imshow(tensor_to_numpy(result))
    ax1.set_title("Result Image")
    plt.show()
