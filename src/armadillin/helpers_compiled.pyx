import numpy as np

def extract_numpy_features(const unsigned char[:] seq,  int[:] mask_positions,  int[:] mask_values):
    cdef int[:] to_output = np.zeros(len(mask_positions),dtype=np.int32)
    cdef int index_mask
    for index_mask in range(len(mask_positions)):
        if seq[mask_positions[index_mask]] == mask_values[index_mask]:
            to_output[index_mask] = 1
    return to_output
