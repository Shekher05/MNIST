import matplotlib.pyplot as plt
from logger import logging
from exception_h import CustomException

def plot_sample_images(X, y, num_images=64, img_shape=(28, 28)):
    """
    Plots a grid of sample images from the dataset.
    """
    try:
        logging.info(f"Plotting {num_images} sample images")
        m = X.shape[0]
        fig, axes = plt.subplots(int(num_images ** 0.5), int(num_images ** 0.5), figsize=(8, 8))
        fig.tight_layout(pad=0.1)
        for i, ax in enumerate(axes.flat):
            if i >= num_images:
                break
            idx = np.random.randint(m)
            ax.imshow(X[idx].reshape(img_shape), cmap='gray')
            ax.set_title(str(y[idx]))
            ax.set_axis_off()
        plt.show()
        logging.info("Sample images plotted")
    except Exception as e:
        logging.error(f"Error plotting images: {e}")
        raise