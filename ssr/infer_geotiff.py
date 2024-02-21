import torch
import argparse
import numpy as np
import rasterio

from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


def prepare_tensor(s2_chunks: np.ndarray, device) -> torch.Tensor:
    """
    Prepare the input tensor for the model.

    Args:
    s2_chunks (np.ndarray): The input data of shape (num_samples, 32, 32, 3)
    device (str): The device to use for the tensor

    Returns:
    torch.Tensor: The input tensor for the model
    """
    # Convert to torch tensor.
    s2_chunks = [torch.as_tensor(img).permute(2, 0, 1) for img in s2_chunks]  # (32, 32, 3) -> (3, 32, 32)
    s2_tensor = torch.cat(s2_chunks).unsqueeze(0)
    s2_tensor = s2_tensor.to(device).float()/255

    # Return input of shape [batch, n_s2_images * channels, 32, 32].
    # Also return an S2 image that can be saved for reference.
    return s2_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help="Path to the options file.")
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load the configuration file.
    opt = yaml_load(args.opt)
    
    geotiff_path = opt['input_geotiff_path']
    upscale_factor = opt['upscale_factor']
    n_lr_images = opt['n_lr_images']  # number of low-res images as input to the model; must be the same as when the model was trained
    save_path = opt['save_path']  # directory where model outputs will be saved
    tile_w = 32
    tile_h = 32

    # Define the generator model, based on the type and parameters specified in the config.
    model = build_network(opt)

    # Load the pretrained weights into the model
    if not 'pretrain_network_g' in opt['path']:
        print("WARNING: Model weights are not specified in configuration file.")
    else:
        weights = opt['path']['pretrain_network_g']  # path to the generator weights
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict[opt['path']['param_key_g']], strict=opt['path']['strict_load_g'])
    model = model.to(device).eval()

    print(f"Running inference on {geotiff_path}")

    # Open the GeoTIFF, get the original data, and set up the upscaled data and transform
    with rasterio.open(geotiff_path) as dataset:
        orig_data = dataset.read()  # (num_samples * 3, h, w)
        orig_crs = dataset.crs

        # create an empty array to store the upscaled data
        upscaled_data = np.zeros((dataset.count, int(dataset.height * upscale_factor), int(dataset.width * upscale_factor)), dtype=np.uint8)

        # scale image transform
        upscaled_transform = dataset.transform * dataset.transform.scale(
            (dataset.width / upscaled_data.shape[-1]), (dataset.height / upscaled_data.shape[-2])
        )
        
    num_possible_samples = orig_data.shape[0] // 3

    assert n_lr_images <= num_possible_samples, f"n_lr_images ({n_lr_images}) must be less than or equal to the number of bands in the input data ({num_possible_samples})."    

    # create a list of random indices to sample from the original data, then grab those samples
    indices = np.sort(np.random.choice(num_possible_samples, n_lr_images, replace=False))

    select_bands = []
    for i in indices:
        select_bands.append(orig_data[i * 3 : (i + 1) * 3])

    model_input_image = np.array(select_bands)  # (num_samples, 3, h, w)
    model_input_image = np.moveaxis(model_input_image, 1, -1)  # (num_samples, h, w, 3)

    # Feed the low-res image tiles through the super-res model
    num_samples, h, w, _ = model_input_image.shape

    # calculate the number of tiles in the x and y directions
    num_tiles_h = h // tile_h
    num_tiles_w = w // tile_w

    # tile the model_input_image and pass each tile to the model
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            tile = model_input_image[:, i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w, :]  # (num_samples, tile_h, tile_w, 3)
            
            input_tensor = prepare_tensor(tile, device)
            output = model(input_tensor)

            output = torch.clamp(output, 0, 1)
            output = output.squeeze().cpu().detach().numpy()
            output = (output * 255).astype(np.uint8)
            
            upscaled_data[
                :,
                i * tile_h * upscale_factor : (i + 1) * tile_h * upscale_factor,
                j * tile_w * upscale_factor : (j + 1) * tile_w * upscale_factor,
            ] = output
    
    with rasterio.open(
        "upscaled.tif",
        "w",
        driver="GTiff",
        width=upscaled_data.shape[2],
        height=upscaled_data.shape[1],
        count=upscaled_data.shape[0],
        dtype=upscaled_data.dtype,
        crs=orig_crs,
        transform=upscaled_transform,
    ) as dst:
        dst.write(upscaled_data)
