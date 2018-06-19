import numpy as np

def gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    if use_label_info:
        pass
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z