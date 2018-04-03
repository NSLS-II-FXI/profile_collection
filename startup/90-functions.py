import numpy as np


def get_img(h, det='Andor'):
    "Take in a Header and return a numpy array of detA1 image(s)."
    img = list(h.data('Andor_image'))
    return np.squeeze(np.array(img))
