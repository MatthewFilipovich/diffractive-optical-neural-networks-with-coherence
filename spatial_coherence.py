import torch


def get_exponentially_decaying_spatial_coherence(field, coherence_degree):
    if coherence_degree < 0 or coherence_degree > 1:
        raise ValueError("Coherence degree must be between 0 and 1.")
    xv, yv = torch.meshgrid(
        torch.arange(field.shape[-1], device=field.device, dtype=torch.double),
        torch.arange(field.shape[-1], device=field.device, dtype=torch.double),
        indexing="ij",
    )
    new_xv = xv - xv[..., None, None]
    new_yv = yv - yv[..., None, None]
    r = torch.sqrt(new_xv**2 + new_yv**2)
    return (field[..., None, None, :, :] * field.conj()[..., None, None]) * coherence_degree**r


def get_source_modes(shape, image_pixel_size):  # shape would be 28 for MNIST
    source_modes = torch.zeros(
        shape**2,  # Number of source modes i.e., total input pixels
        shape * image_pixel_size,  # Nx
        shape * image_pixel_size,  # Ny
        dtype=torch.cdouble,
    )
    for i in range(shape):
        for j in range(shape):
            source_modes[
                i * shape + j,
                i * image_pixel_size : (i + 1) * image_pixel_size,
                j * image_pixel_size : (j + 1) * image_pixel_size,
            ] = 1
    return source_modes
