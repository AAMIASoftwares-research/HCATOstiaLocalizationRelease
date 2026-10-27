import warnings
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
from typing import Literal
import os, time
import math
import itertools
import numpy
import torch
from monai.inferers import sliding_window_inference
from skimage.measure import marching_cubes

from hcatostialocalization.model.dsnt_layer import DSNT3D
from hcatostialocalization.logger.stdout import log, debug, error



def get_location_from_probability_one_channel(heatmap, method: Literal['centroid', 'dsnt']='centroid', window_size=11) -> torch.Tensor:
    """
    Finds maximum peak of the heatmap and calculates centroid 
    only in a small local window for sub-pixel precision.

    Input:
    heatmap: torch.Tensor|monai MetaTensor of shape (R, A, S) only space axes.

    Returns:
    torch.tensor of shape (3,)
    """
    match method:
        case 'centroid':
            max_idx = torch.argmax(heatmap)
            D, H, W = heatmap.shape
            
            z_peak = (max_idx // (H * W)).item()
            y_peak = ((max_idx % (H * W)) // W).item()
            x_peak = (max_idx % W).item()
            
            half_w = window_size // 2
            z_min, z_max = max(0, z_peak - half_w), min(D, z_peak + half_w + 1)
            y_min, y_max = max(0, y_peak - half_w), min(H, y_peak + half_w + 1)
            x_min, x_max = max(0, x_peak - half_w), min(W, x_peak + half_w + 1)
            
            local_window = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
            
            # Subtract minimum of the local window to zero out baseline noise
            # and guarantee always non-negative weights well-shaped around the peak.
            local_window = local_window - local_window.min()
            
            sum_weights = torch.sum(local_window) + 1e-6
            
            z_coords = torch.arange(local_window.shape[0], device=heatmap.device, dtype=torch.float32)
            y_coords = torch.arange(local_window.shape[1], device=heatmap.device, dtype=torch.float32)
            x_coords = torch.arange(local_window.shape[2], device=heatmap.device, dtype=torch.float32)
            
            grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            
            c_z = torch.sum(local_window * grid_z) / sum_weights
            c_y = torch.sum(local_window * grid_y) / sum_weights
            c_x = torch.sum(local_window * grid_x) / sum_weights
            
            final_z = z_min + c_z.item()
            final_y = y_min + c_y.item()
            final_x = x_min + c_x.item()
            location_tensor = torch.tensor([final_z, final_y, final_x], dtype=torch.float32)
        case 'dsnt':
            dsnt_layer = DSNT3D(normalized_coords=True)
            pred_coords = dsnt_layer.get_coords_from_probs(heatmap[None, None, ...])
            location_tensor = pred_coords.to(torch.float32).flatten()
        case _:
            raise ValueError("method argument not recognized.")
    return location_tensor




def predict_minibatches_inplace(
    model: torch.nn.Module,
    # image related params
    images: torch.Tensor,
    patch_size: int | tuple[int,...],
    overlap: int | tuple[int,...] = 0,
    minibatch_size: int = 1,
    ensure_sdims_shape_multiple_of: int | tuple[int] | None = None,
    # model related params
    activation: torch.nn.Sigmoid | torch.nn.Softmax | torch.nn.Module | None = None,
    augment: bool = False,
    model_features_to_save: list[str] | None = None,
    # device and memory params
    output_logits_dtype: torch.dtype = torch.float32, # if possible, use torch.bfloat16
    storage_device_output: torch.device | str | None = None,
    storage_device_features: torch.device | str | None = None,
    # fallback to MONAI sliding window inference
    use_monai: bool = True,
    # logging
    verbose: bool = False,
) -> dict:
    """
    Memory-efficient sliding-window inference with optional TTA.
    
    Allows saving and returning arbitrary intermediate features from the model,
    provided that ther model has the required hooks set up:
    
    ````
    features:   model.<feature_name>
    flag:       model.save_<feature_name>
    ````

    If flag is true, hook must save features inside model.<features_name> as tensors
    with same ndim and batch dimension as input tensor (of course channel and spatial dimensions can vary).
    A list of <feature_name>s can be passed via the `model_features_to_save: list[str]` argument, and this
    function will assume the behaviur above.

    ATTENTION: you have to set model.save_<feature_name> to True before passing the model to this method.
    After the method, remember to set it to False again if need be.
    
    Upsamples multi-scale features to patch-size on the fly, allowing 
    all logits and features to share the exact same accumulation logic.

    ATTENTION: Accumulating features will drain your memory. For example, an input of shape (1, 1, 200, 500, 500)
    will produce features, assuming features size (channels) 512 which is common, of 95GB in float32, 
    and 47GB in float16. Use this feature only if your machine can support this kind of resources.

    # Args

    - model: the model to use (must be a child of torch.nn.Module)
    - images: a torch.Tensor of shape (B, C, K1, K2) or (B, C, K1, K2, K3) with C compatible with model input
    - patch_size: int or tuple of same len as spatial shape of image (Kn). The patches in which to subdivide the image
      to perform windowed prediction.
    - overlap: int or tuple of same len as spatial shape of image (Kn). number of pixel/voxels that overlap in windowed prediction.
    - minibatch_size: number of samples to feed the model (low if big images and big model).
    - ensure_sdims_shape_multiple_of: medical images can be of any size and shape, but models usually have some basic measures they should comply with.
      most models should have input image shape that is a multiple of 16, 32, or 64. This parameter makes the image compatible with the model
      required constraint under the hood. The final segmentation mask spatial shape will be the same as the one of the input image, not the one of the
      corrected one. If unsure, set it to 32 or 16.
    - activation: the activation function to apply. If none, it is assumed that the model already applied the activation itself. 
      Else, choose between softmax and sigmoid.
    - augment: if truue, enables standard test-time augmentations. Prediction will take longer, but final result should be better.
    - model_features_to_save: a list of names of models features to be saved. See above.
    - output_logits_dtype: the dtype of the output logits, use float16 or better bfloat16 to save memory.
    - storage_device_output: where to store the output segmentation (cpu or gpu).
    - storage_device_features: where to store features tensor (usually they can get massive). If None, 'cpu' is assumed
    - use_monai (bool = False): use MONAI's sliding_window_inference.
    - verbose: if true, prints comments about this function execution.
    
    # Returns
    A dict[str: torch.Tensor] with keys:
    - 'prediction'
    - 'features'
    - 'overlap_mask' (int16, same device as predictions)

    """

    # #################### #
    # MONAI IMPLEMENTATION #
    # #################### #
    if use_monai:
        if verbose:
            print("Executing predict_minibatches_inplace fallback: MONAI sliding_window_inference.")
        spatial_dims = images.ndim - 2
        p_size = (patch_size,) * spatial_dims if isinstance(patch_size, int) else tuple(patch_size)
        o_size = (overlap,) * spatial_dims if isinstance(overlap, int) else tuple(overlap)
        overlap_ratios = tuple(o / p for o, p in zip(o_size, p_size))
        max_overlap_ratio = max(overlap_ratios) if overlap_ratios else 0.0 # MONAI preference to have just one overlap value
        if augment and verbose:
            print("Warning: 'augment=True' requested, but standard MONAI sliding window does not natively perform TTA.")
        if model_features_to_save and verbose:
            print(f"Warning: model features {model_features_to_save} will NOT be extracted in MONAI fallback mode.")
        out_device = storage_device_output if storage_device_output is not None else images.device
        # pad to satisfy model input image dimensions requirements if specified
        pad_needed = False
        padded_patch_size = list(p_size)
        n_sdim: int = images.ndim - 2
        if ensure_sdims_shape_multiple_of is not None:
            if isinstance(ensure_sdims_shape_multiple_of, int):
                ensure_sdims_shape_multiple_of = (ensure_sdims_shape_multiple_of,) * n_sdim
            elif isinstance(ensure_sdims_shape_multiple_of, tuple):
                diff = max(0, n_sdim - len(ensure_sdims_shape_multiple_of))
                ensure_sdims_shape_multiple_of = (0,) * diff + ensure_sdims_shape_multiple_of # + in tuples extends the tuple
            pad_needed = any(s > 0 for s in ensure_sdims_shape_multiple_of)
            if pad_needed:
                inference_model = PaddingWrapper(model, ensure_sdims_shape_multiple_of)
            else:
                inference_model = model
        else:
            inference_model = model
        # mode
        is_full_image = all(p >= s for p, s in zip(p_size, images.shape[2:]))
        active_mode = "constant" if is_full_image else "gaussian"
        # monai inference
        with torch.inference_mode():
            logits_out = sliding_window_inference(
                inputs=images,
                roi_size=p_size,
                sw_batch_size=minibatch_size,
                predictor=inference_model,
                overlap=max_overlap_ratio,
                mode=active_mode,
                device=images.device,
                sw_device=images.device,
                progress=verbose
            )
        logits_out = logits_out.to(dtype=output_logits_dtype, device=out_device)
        if activation is not None:
            logits_out = activation(logits_out)
        return {
            'prediction': logits_out,
            'features': None,
            'overlap_mask': None
        }
    

    # ##################### #
    # CUSTOM IMPLEMENTATION #
    # ##################### #

    if not (images.ndim in [4, 5]):
        raise ValueError(f"Argument images must be a tensor of shape (B,C,K1, K2 [, K3]) with 4 or 5 dimensions, got: {images.ndim}")

    n_sdim: int = images.ndim - 2
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * n_sdim
    if len(patch_size) != n_sdim:
        raise ValueError(f"Argiument patch_size must be an integer or a tuple of length == (images.shape - 2) [in this case {n_sdim}]. Got {patch_size} (len: {len(patch_size)})")
    if isinstance(overlap, int): 
        overlap = (overlap,) * n_sdim
    if len(overlap) != n_sdim:
        raise ValueError(f"Argiument overlap must be an integer or a tuple of length == (images.shape - 2) [in this case {n_sdim}]. Got {overlap} (len: {len(overlap)})")

    # 0. Check image sizes and patch size, and pad image if patch size is bigger than image size on some dimension
    # If the image is smaller than the patch size, pad it with zeros safely.
    orig_spatial_shape = images.shape[2:]
    pad_needed_for_patch = False
    pad_list = []
    # torch.nn.functional.pad expects padding definitions from the last dimension moving forward
    for i in range(n_sdim - 1, -1, -1): # e.g. [2, 1, 0]
        diff = max(0, patch_size[i] - images.shape[2 + i])
        pad_list.extend([0, diff])  # pad right/bottom/back (keep top-left origin same)
        if diff > 0:
            pad_needed_for_patch = True
    if pad_needed_for_patch:
        images = torch.nn.functional.pad(images, pad_list, mode='constant', value=images.min().item())


    # 1. Generate patch indices
    starts = []
    import itertools
    for i, dim in enumerate(range(2, images.ndim)):
        step = max(1, patch_size[i] - overlap[i])
        start = list(range(0, max(images.shape[dim] - patch_size[i] + 1, 1), step))
        r = images.shape[dim] - (start[-1] + patch_size[i])
        left = (r // 2) + (r % 2)
        start = [s + left for s in start]
        start = [0] + start if 0 not in start else start
        start = start + [images.shape[dim] - patch_size[i]] if images.shape[dim] - patch_size[i] not in start else start
        starts.append(start)

    patch_idx_list = []
    if n_sdim == 2:
        for s in itertools.product(*starts):
            patch_idx_list.append([[s[0], s[0]+patch_size[0]], [s[1], s[1]+patch_size[1]]])
    else:
        for s in itertools.product(*starts):
            patch_idx_list.append([[s[0], s[0]+patch_size[0]],
                                   [s[1], s[1]+patch_size[1]],
                                   [s[2], s[2]+patch_size[2]]])

    # 2. Compute MASTER spatial normalization mask (shared for both logits and features)
    mask_shape = (1, 1) + images.shape[2:]
    overlap_mask = torch.zeros(mask_shape, dtype=output_logits_dtype, device=storage_device_output)
    for pidx in patch_idx_list:
        sl = [slice(None), slice(None)] + [slice(pidx[d][0], pidx[d][1]) for d in range(n_sdim)]
        overlap_mask[tuple(sl)] += 1
    overlap_mask = torch.clamp(overlap_mask, min=1.0)

    # 3. Handle padding wrapper safely
    pad_needed = False
    padded_patch_size = list(patch_size)
    if ensure_sdims_shape_multiple_of is not None:
        if isinstance(ensure_sdims_shape_multiple_of, int):
            ensure_sdims_shape_multiple_of = (ensure_sdims_shape_multiple_of,) * n_sdim
        elif isinstance(ensure_sdims_shape_multiple_of, tuple):
            diff = max(0, n_sdim - len(ensure_sdims_shape_multiple_of))
            ensure_sdims_shape_multiple_of = (0,) * diff + ensure_sdims_shape_multiple_of
            
        pad_needed = any(s > 0 for s in ensure_sdims_shape_multiple_of)
        if pad_needed:
            inference_model = PaddingWrapper(model, ensure_sdims_shape_multiple_of)
            for i, size in enumerate(patch_size):
                m = ensure_sdims_shape_multiple_of[i]
                pad_amt = (m - size % m) % m if m > 0 else 0
                padded_patch_size[i] = size + pad_amt
        else:
            inference_model = model
    else:
        inference_model = model

    if storage_device_output is None:
        storage_device_output = next(model.parameters()).device
    storage_device_output = torch.device(storage_device_output)

    # 4. Dummy shape probe & Pre-allocation
    dummy_spatial = torch.zeros((1, images.shape[1], *patch_size), device=images.device)
    with torch.inference_mode():
        dummy_out = inference_model(dummy_spatial)
        
    out_spatial = tuple(images.shape[2:]) 
    out_ch = dummy_out.shape[1]
    
    logits_out = torch.zeros((images.shape[0], out_ch) + out_spatial, dtype=output_logits_dtype, device=storage_device_output)

    features_out = {}
    if model_features_to_save:
        for k in model_features_to_save:
            f_tmp = getattr(model, k, None)
            if f_tmp is not None and torch.is_tensor(f_tmp):
                # Features are now allocated at the exact same full-image resolution as logits
                features_out[k] = torch.zeros((images.shape[0], f_tmp.shape[1]) + out_spatial, dtype=output_logits_dtype, device=storage_device_features)
            else:
                features_out[k] = None

    # 5. TTA transforms
    tta_transforms = []
    if n_sdim == 2:
        tta_transforms = [
            # (transform, go back to original)
            (lambda x: x, lambda x: x), # first one must be identity
            (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
            (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
            (lambda x: torch.rot90(x, k=1, dims=[-2, -1]), lambda x: torch.rot90(x, k=3, dims=[-2, -1])),
            (lambda x: torch.rot90(x, k=2, dims=[-2, -1]), lambda x: torch.rot90(x, k=2, dims=[-2, -1])),
            (lambda x: torch.rot90(x, k=3, dims=[-2, -1]), lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
        ]
    else:
        tta_transforms = [
            # (transform, go back to original)
            (lambda x: x, lambda x: x), # first one must be identity
            (lambda x: torch.flip(x, dims=[-3]), lambda x: torch.flip(x, dims=[-3])),
            (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
            (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
            (lambda x: torch.flip(x, dims=[-3, -2]), lambda x: torch.flip(x, dims=[-3, -2])),
            (lambda x: torch.flip(x, dims=[-3, -1]), lambda x: torch.flip(x, dims=[-3, -1])),
            (lambda x: torch.flip(x, dims=[-2, -1]), lambda x: torch.flip(x, dims=[-2, -1])),
            (lambda x: torch.flip(x, dims=[-3, -2, -1]), lambda x: torch.flip(x, dims=[-3, -2, -1])),
        ]
    n_tta = len(tta_transforms) if augment else 1

    # 6. Process patches in chunks
    B = images.shape[0]
    
    # --- FLATTEN THE TASK LIST TO ENSURE STRICT BATCH SIZES ---
    # Instead of looping over images (Batch B) and then looping over spatial patches inside that, 
    # the code flattens everything into a single 1D queue of "tasks." 
    # Each task is just a tuple containing (which_image, which_spatial_patch).
    all_tasks = [(b, pidx) for b in range(B) for pidx in patch_idx_list]
    n_tasks = len(all_tasks)

    if verbose:
        log(f"Processing {n_tasks} total sub-volumes (B={B} x {len(patch_idx_list)} spatial patches) in chunks of {minibatch_size}..." + (" [TTA]" if augment else ""))
        t0 = time.time()
    
    # The loop steps through the all_tasks queue, grabbing exactly minibatch_size tasks at a time. 
    # If minibatch_size=4, it grabs 4 patch coordinates (regardless of whether they belong to the 
    # same image or different images in the batch).
    tasks_iterator = list(range(0, n_tasks, minibatch_size))
    for i in tasks_iterator:
        chunk_tasks = all_tasks[i:i+minibatch_size]
        
        # For the current chunk of 4 patches, the code enters the Test-Time Augmentation (TTA) loop. 
        # It grabs a pair of lambda functions:
        #   inp_fn: Modifies the input (e.g., flips it horizontally).
        #   inv_fn: Reverses the modification on the output (e.g., flips it horizontally again to put it back to normal).
        # It user does not require TTA, n_tta = 1, so only the first TTA in the list is executed, which is the identity transformation (lambda x: x)
        for tta_idx in range(n_tta):
            inp_fn, inv_fn = tta_transforms[tta_idx]

            # This loop extracts <minibatch_size> patches from the input images tensor, augments it with the current TTA, and
            # creates a list of patches that will become the input of the model
            chunk_imgs = []
            for (b, pidx) in chunk_tasks:
                # Extract the 1 x C x K1 x K2 patch, preserving batch dim size 1
                p_sl = tuple([slice(b, b+1), slice(None)] + [slice(pidx[d][0], pidx[d][1]) for d in range(n_sdim)])
                p = images[p_sl]
                chunk_imgs.append(inp_fn(p) if inp_fn else p)

            # Concatenate along batch dimension -> Shape guaranteed to be: (len(chunk_tasks), C, *spatial)
            # This is the input of the model
            chunk_tensor = torch.cat(chunk_imgs, dim=0)

            with torch.inference_mode():
                # Model strictly receives at most minibatch_size items
                logits_chunk = inference_model(chunk_tensor)

            # Accumulate (Vectorized across the exact tasks processed)
            # Now that the model has predicted masks for the <minibatch_size> patches (say, 4 patches), we have to put them back into the giant logits_out canvas.
            #    Targeting: It recalculates the exact same slice sl used to extract the patch, but now uses it to target the destination coordinate in logits_out.
            #    Reversal: It applies inv_fn to transform the prediction back to its original geometric orientation.
            #    Safeguards: It clamps the raw linear logits between -1000 and 1000 and converts any NaN values to 0.0. This prevents half-precision (float16/bfloat16) overflow explosions during summation.
            #    Addition: It adds the patch into the master canvas. Because patches overlap, the values physically add together (which is why Step 7 later divides by the overlap_mask*n_tta).
            for j, (b, pidx) in enumerate(chunk_tasks):
                # Target slice in the master output tensor
                sl = [slice(b, b+1), slice(None)] + [slice(pidx[d][0], pidx[d][1]) for d in range(n_sdim)]
                # Extract the patch for THIS specific task -> Shape: (1, C, *spatial)
                out_patch = logits_chunk[j:j+1]
                # Accumulate Logits (do not apply activation, it is applied at the end)
                acc = inv_fn(out_patch) if inv_fn else out_patch
                acc = torch.clamp(acc, -1e3, 1e3)
                acc = torch.nan_to_num(acc, nan=0.0)
                # accumulate into the main logits_out
                # Why do we accumulate logits and not probabilities?
                #   For overlapping patches and TTA (Internal): We are averaging outputs from the exact same model. The scale is identical, so accumulating raw logits 
                #                                               avoids the mathematical distortion of averaging non-linear curves.
                #   For Ensembling (External):                  We are averaging outputs from different models. The scales are mismatched, so you must use probabilities 
                #                                               to normalize their "voting power" (soft voting).
                logits_out[tuple(sl)] += acc.to(device=storage_device_output, dtype=output_logits_dtype)

                # Accumulate Features
                # This final block of the loop does the exact same accumulation process as for logits_out, but targets the intermediate features saved by the model hooks.
                # Because intermediate features are usually at a lower resolution than the output logits (e.g., a UNet bottleneck), this block includes logic to 
                # torch.nn.functional.interpolate the feature patch up to the patch_size before applying the geometric reversal and adding it to the features_out canvas.
                # This might seem a waste of memory, however it is necessary to seamlessly align features back together in a unified space.
                if model_features_to_save:
                    for k in model_features_to_save:
                        if features_out[k] is not None:
                            feat = getattr(model, k, None)
                            if feat is not None:
                                # Extract feature patch for THIS specific task -> Shape: (1, C_feat, *spatial)
                                feat_patch = feat[j:j+1].float() 
                                
                                if pad_needed:
                                    feat_upsampled = torch.nn.functional.interpolate(
                                        feat_patch,
                                        size=tuple(padded_patch_size),
                                        mode='nearest'
                                    )
                                    # Crop to original patch
                                    crop_slices = [slice(None), slice(None)] + [slice(0, ps) for ps in patch_size]
                                    feat_upsampled = feat_upsampled[tuple(crop_slices)]
                                else:
                                    feat_upsampled = torch.nn.functional.interpolate(
                                        feat_patch,
                                        size=patch_size,
                                        mode='nearest'
                                    )
                                    
                                feat_acc = inv_fn(feat_upsampled) if inv_fn else feat_upsampled
                                features_out[k][tuple(sl)] += feat_acc.to(device=storage_device_features, dtype=output_logits_dtype)

    # 7. Final normalization
    # if no augment, n_tta = 1
    logits_out /= (n_tta * overlap_mask)
    
    # Note: Softmax and Sigmoid inherit from torch.nn.Module, so we can just check for Module or None safely.
    if activation is not None:
        if isinstance(activation, torch.nn.Sigmoid):
            if verbose:
                log("Applying Sigmoid in-place to save memory...")
            logits_out.sigmoid_()  # Strictly in-place, zero memory overhead
        else:
            # Softmax cannot be done strictly in-place in PyTorch
            # Fallback for custom modules (ReLU, SiLU, etc.)
            logits_out = activation(logits_out)

    # Features use the exact same normalization mask as logits now
    overlap_mask = overlap_mask.to(device=storage_device_features, dtype=output_logits_dtype)
    for k in features_out:
        if features_out[k] is not None:
            features_out[k] = features_out[k].to(device=storage_device_features, dtype=output_logits_dtype)
            features_out[k] /= (n_tta * overlap_mask)

    # Crop the padded regions off to match the original user input exactly
    if pad_needed_for_patch:
        crop_slices = [slice(None), slice(None)] + [slice(0, s) for s in orig_spatial_shape]
        crop_slices = tuple(crop_slices)
        logits_out = logits_out[crop_slices]
        for k in features_out:
            if features_out[k] is not None:
                features_out[k] = features_out[k][crop_slices]

    if verbose:
        minutes, seconds = divmod(time.time() - t0, 60)
        log(f"predict_minibatches_inplace: done in {int(minutes)}m {int(seconds):02d}s")
    
    # put mask in int16 and device = image device
    overlap_mask = overlap_mask.to(dtype=torch.int16, device=logits_out.device)

    out = {
        'prediction': logits_out,
        'features': features_out,
        'overlap_mask': overlap_mask
    }
    return out





def compute_pca_features(
    features: torch.Tensor,
    pca_components: slice|list[int] = slice(0,3),
    whitening: bool = False,
    memory_cap_mb: float = 100.0,
    compute_device: torch.device | str | None = None
) -> dict[str, torch.Tensor]:
    """
    Computes a memory-efficient 3-component PCA on a high-dimensional spatial tensor.
    
    Args:
        features: torch.Tensor of shape (1, C, *spatial_dimensions) (e.g., 1, C, D, H, W)
        pca_components: slice or list of indices, where 0 indicates the "strongest" component. Default: top 3.
        whitening: bool that, if true, applies whitening and removes outliers.
        memory_cap_mb: Maximum memory in MB to allocate for chunked operations.
        compute_device: The device to perform the matrix multiplications on (usually 'cuda').
        
    Returns:
        dict containing:
        - 'pca'
        - 'eigenvalues'
        - 'eigenvectors'
        - 'covariance_matrix'
        - 'features_mean'
        all matching the input tensor's dtype and device. 
        'pca' will have shape (1, n_components, *spatial_dimensions) and values in range [0.0, 1.0].
        'eigenvalues' and 'eigenvectors' are exactly how they are obtained from the covariance matrix.
        For plotting, you have to reverse them:
        ```
        evals = torch.flip(eigenvalues, dims=[0])
        evecs = torch.flip(eigenvectors, dims=[1])
        ```
        For plotting pca, convert to (*spatial_dimensions, n_components) with torch.movedims(input[None], source, destination) and pass to numpy float16.
    """
    if compute_device is None:
        compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        compute_device = torch.device(compute_device)

    orig_dtype = features.dtype
    orig_device = features.device

    features = features[0] # (c, ...)
    C = features.shape[0]
    spatial_shape = features.shape[1:]
    
    # slicing
    if isinstance(pca_components, slice):
        _start, _stop, _step = pca_components.indices(C)
        pca_components = list(range(_start, _stop, _step))
    if isinstance(pca_components, tuple):
        pca_components = list(pca_components)
    pca_components_from_end = [-1-a for a in pca_components]

    # Fallback for small feature counts (<= 3)
    if C < len(pca_components):
        pad_size = len(pca_components) - C
        padding = torch.zeros((pad_size, *spatial_shape), dtype=orig_dtype, device=orig_device)
        features = torch.cat([features, padding], dim=0)
        
    # Flatten spatial dimensions -> Shape: (N, C)
    flat_features = features.reshape(C, -1).t()
    total_voxels = flat_features.shape[0]

    # Calculate chunk size dynamically based on memory cap
    # 1 float32 takes 4 bytes. In the loop, we hold the chunk, its centered version,
    # and intermediate matmul outputs. We multiply by 3 to ensure a generous safety margin.
    bytes_per_voxel_operation = C * 4 * 3 
    chunk_size = max(1, int((memory_cap_mb * 1024 * 1024) / bytes_per_voxel_operation))
    chunk_size = int(min(chunk_size, flat_features.shape[1]//4))

    # --- 1. Global Mean Centering ---
    feat_mean = torch.zeros(C, dtype=torch.float32, device=compute_device)
    for start_idx in range(0, total_voxels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_voxels)
        chunk_sum = flat_features[start_idx:end_idx].sum(dim=0, dtype=torch.float32)
        feat_mean += chunk_sum.to(dtype=torch.float32, device=compute_device)
    feat_mean /= total_voxels

    # --- 2. Iterative Covariance Matrix ---
    cov_matrix = torch.zeros((C, C), dtype=torch.float32, device=compute_device)
    for start_idx in range(0, total_voxels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_voxels)
        chunk = flat_features[start_idx:end_idx].to(dtype=torch.float32, device=compute_device)
        chunk.sub_(feat_mean)
        cov_matrix += torch.matmul(chunk.t(), chunk)

    # --- 3. Eigendecomposition ---
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    # Extract the top 3 components (last 3 columns) -> Shape: (C, 3)
    V = eigenvectors[:, pca_components_from_end]
    # Flip so the most important component is first
    V = torch.flip(V, dims=[1]).to(dtype=torch.float32, device=compute_device)
    # whitening
    if whitening:
        evals = eigenvalues[pca_components_from_end]
        evals = torch.flip(evals, dims=[0]).to(dtype=torch.float32, device=compute_device)
        whiten_scale = 1.0 / torch.sqrt(evals + 1e-5)

    # --- 4. Projection ---
    pca_features = torch.empty((total_voxels, len(pca_components)), dtype=orig_dtype, device=orig_device)

    for start_idx in range(0, total_voxels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_voxels)
        chunk = flat_features[start_idx:end_idx].to(dtype=torch.float32, device=compute_device)
        chunk.sub_(feat_mean)
        proj_chunk = torch.matmul(chunk, V)
        # whitening
        if whitening:
            proj_chunk *= whiten_scale
        pca_features[start_idx:end_idx] = proj_chunk.to(dtype=orig_dtype, device=orig_device)

    # --- 5. Formatting & Normalization ---
    # Normalize the RGB array to [0, 1] across all channels/voxels
    if whitening:
        pca_features = pca_features.to(torch.float32)
        # we have to subsample the features, otherwise torch cries. 
        # since features are usually upsampled to be of same shape of input, but really are
        # blocks of 16 or even 32, 64 and within they are all the same, we try with 16, and it not enough we move to 32
        _f = torch.movedim(pca_features, source=-1, destination=0)[None, ...] # (spatial, Nf) -> (1, Nf, spatial)
        _success = False
        _exp = 4 # 2^4 = 16
        while not _success:
            _step = 2**_exp
            slices = [0, slice(None)] + [slice(_step,_f.shape[i]-_step,_step) for i in range(2, _f.ndim)]
            _fs = _f[*slices] # subsample with logic
            _fs = _fs.reshape(len(pca_components), -1).t() # (n_subsampled, N_pca_features)
            _success = True if _fs.shape[0] < 10000000 else False # 2^24=16777216 torch C++ limit, plus a good margin for performance
            _exp += 1
    feat_min = torch.quantile(_fs, 0.01, dim=0) if whitening else pca_features.min()
    feat_max = torch.quantile(_fs, 0.99, dim=0) if whitening else pca_features.max()
    pca_features -= feat_min
    pca_features /= (feat_max - feat_min + 1e-8)
    pca_features.clamp_(min=0.0, max=1.0)
    if whitening:
        pca_features = pca_features.to(orig_dtype)
    pca_features = torch.movedim(pca_features, source=-1, destination=0)[None, ...] # spatial, Nf -> 1, Nf, spatial
    pca_features = pca_features.reshape(1, len(pca_components), *spatial_shape)

    # Format full eigenvalues/eigenvectors in descending order for plotting
    eigenvalues = eigenvalues.to(dtype=orig_dtype, device=orig_device)
    eigenvectors = eigenvectors.to(dtype=orig_dtype, device=orig_device)
    cov_matrix = cov_matrix.to(dtype=orig_dtype, device=orig_device)
    feat_mean = feat_mean.to(dtype=orig_dtype, device=orig_device)

    return {
        'pca': pca_features,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'covariance_matrix': cov_matrix,
        'features_mean': feat_mean
    }












def predict_minibatches(
    model: torch.nn.Module,
    activation: torch.nn.Sigmoid | torch.nn.Softmax | torch.nn.Module,
    images: torch.Tensor,
    minibatch_size: int,
    model_features_to_save: list[str] = [],
    augment: bool = False,
    ensure_sdims_shape_multiple_of: int|tuple[int]|None = None,
    output_logits_dtype: torch.dtype = torch.float32,
    storage_device: torch.device | str | None = None,
    verbose: bool = False
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """
    Returns logits and features for a set of images.

    # Args
    - model: the model to use for inference.
    - images: input images of shape (N, C, ...).
    - minibatch_size: size of the minibatches to use for inference. Recommended: 32 or lower (16 or lower if `augmentation=True`)
    - model_features_to_save: list of attributes of the model to save as features as strings.
    - augment: (bool, default: False) whether to apply test-time augmentation such as flips and rotations
               to compose the final logits as the median of the ensamble of predictions. Slice-by-slice only.
    - ensure_sdims_shape_multiple_of: (int|tuple[int]|None, default: None) ensures that the spatial dimensions 
               of the input images are multiples of the specified value(s) by padding them before inference.
    - output_logits_dtype: (torch.dtype, default: torch.float32) the data type of the output logits tensor.
    - storage_device: (torch.device|str|None, default: None) the device where the output logits and features 
               will be stored. If None, the device of the model's first output is used.
    - verbose: (bool, default: False) whether to print progress.

    # Returns
    - logits: (torch.Tensor) the output of the model for each input image (same batch dimension as `images`).
    - features: a dictionary with keys the elements of `model_features_to_save` and values
      the corresponding features concatenated along the batch dimension.
    """
    # Minibatches to keep memory low for bigger models
    # If some features are to be saved, model must already have
    #  the corresponding switch on (e.g. model.save_features = True)
    if not images.ndim in [4, 5]:
        raise ValueError(f"Images should be 4D (2d images) or 5D (3d volumes) tensors, got {images.ndim}D tensor instead.")
    image_geometry = '2d' if images.ndim == 4 else '3d'
    if ensure_sdims_shape_multiple_of is not None:
        if isinstance(ensure_sdims_shape_multiple_of, int):
            ensure_sdims_shape_multiple_of = (ensure_sdims_shape_multiple_of,) * (images.ndim-2)
        if isinstance(ensure_sdims_shape_multiple_of, tuple):
            # starting from the last dimension, make the tuple as long as ndims-2 by adding 0 (the wildcard for "no change") at the beginning if needed
            ensure_sdims_shape_multiple_of = (0,) * max(0, (images.ndim-2) - len(ensure_sdims_shape_multiple_of)) + ensure_sdims_shape_multiple_of
        if len(ensure_sdims_shape_multiple_of) != (images.ndim-2):
            raise ValueError(f"Length of ensure_sdims_shape_multiple_of tuple should be equal to the number of spatial dimensions of the images (ndims-2), got {len(ensure_sdims_shape_multiple_of)} but expected {images.ndim-2}.")
    else:
        ensure_sdims_shape_multiple_of = (0,) * (images.ndim-2)

    # if we must ensure that the spatial dimension of the images that the model take as input,
    # we can pad the images using this wrapper class
    if any([s > 0 for s in ensure_sdims_shape_multiple_of]):
        inference_model = PaddingWrapper(model, ensure_sdims_shape_multiple_of)
    else:
        inference_model = model

    # dummy run - needed for memory allocation
    _ds_slices = tuple([slice(0, 1), slice(None)] + [slice(0, s) for s in images.shape[2:]])
    with torch.inference_mode():
        logits: torch.Tensor = inference_model(images[_ds_slices])
        if storage_device is None:
            storage_device = logits.device

    # Initialize the final logits tensor - saves memory and faster
    logits_val = torch.zeros(
        [images.shape[0]]+[logits.shape[_s] for _s in range(1, logits.ndim)],
        dtype=output_logits_dtype,
        device=storage_device
    ) 

    # Initialize features
    _unknown_attr = [f for f in model_features_to_save if not hasattr(model, f)]
    if len(_unknown_attr) != 0:
        raise KeyError(f"Model has no attribute(s) {_unknown_attr}")
    
    features = {
        # initialize the feaures tesor of the batch
        # After the dummy run, if the model is set to save features,
        # there should already be a tensor in the required features, and not be empty (None).
        # If it is None (empty, non-tensor), then we initialize it as None, and it will be ignored
        # in the followig logic
        k: torch.zeros(
            [images.shape[0]]+[getattr(model, k).shape[_s] for _s in range(1, getattr(model, k).ndim)],
            dtype=output_logits_dtype,
            device=storage_device
        ) if torch.is_tensor(getattr(model, k)) else None
        for k in model_features_to_save
    }

    # Run loop
    if verbose:
        log(f"Running inference in minibatches of size {minibatch_size} for {images.shape[0]} images...")
        _t0 = time.time() 
    
    for i in range(0, images.shape[0], minibatch_size):
        if verbose:
            print(f"  {100.0 * i / images.shape[0]: 3.2f}% done", end='\r')
        max_i = min(i+minibatch_size, images.shape[0])
        if i == max_i:
            # this works because i:i+1 returns a tensor with one element
            max_i += 1
        # save logits for evaluation outside the scope of this fucntion
        if not augment:
            logits_val[i:max_i] += inference_model(images[i:max_i]).to(storage_device)
        else:
            # TEST TIME AUGMENTATIONS #
            # ####################### #
            if image_geometry == '2d':
                # Initialize "Winsorized Running Sum" -> mean without extreme outliers
                # wrs_fn = lambda x: torch.clip(x, -10.0, 10.0) # <--- THIS WAS REMOVED
                # The wrs was removed because:
                # 1. **Jensen's Inequality**: `mean(softmax(x)) ≠ softmax(mean(x))`. Averaging logits penalizes uncertainty more heavily than averaging probabilities. This
                #   systematically shifts the decision boundary toward higher confidence regions, often hurting generalization on ambiguous tissue boundaries.
                # 2. **Clipping breaks relative margins**: If augmentation A gives logits `[15, 2]` and augmentation B gives `[12, 8]`, your code clips them to `[10, 2]` and `[10,
                #   8]`, then averages to `[10, 5]`. Original probabilities averaged would be `mean([0.99995, 0.00005], [0.9997, 0.0003]) ≈ [0.9998, 0.0002]`. Clipping artificially
                #   flattens the margin, reducing TTA's ability to resolve conflicts between augmentations.
                # 3. **TTA is meant to average calibrated confidences**, not clamp raw scores. The `wrs_fn` (Winsorized Running Sum) name suggests you wanted to dampen outliers,
                #   but applying it per-augmentation before accumulation defeats the statistical purpose of TTA.
                '''
                _aug_counter = 0
                wrs_fn = lambda x: torch.nan_to_num(x, nan=0.0, posinf=12.0, neginf=-12.0)
                # K1 flip
                logits_val[i:max_i] += torch.flip(
                    wrs_fn(inference_model(
                        torch.flip(images[i:max_i], dims=[-2])
                    )),
                    dims=[-2]
                ).to(storage_device)
                _aug_counter += 1
                # K2 flip
                logits_val[i:max_i] += torch.flip(
                    wrs_fn(inference_model(
                        torch.flip(images[i:max_i], dims=[-1])
                    )),
                    dims=[-1]
                ).to(storage_device)
                _aug_counter += 1
                # K1+K2 flip == 180° rotation in 2D
                logits_val[i:max_i] += torch.flip(
                    torch.flip(
                        wrs_fn(inference_model(
                            torch.flip(
                                torch.flip(images[i:max_i], dims=[-2]),
                                dims=[-1]
                            )
                        )),
                        dims=[-1]
                    ),
                    dims=[-2]
                ).to(storage_device)
                _aug_counter += 1
                # rotation 90 deg
                logits_val[i:max_i] += torch.rot90(
                        wrs_fn(inference_model(
                            torch.rot90(images[i:max_i], k=1, dims=[-2, -1])
                        )),
                        k=3,
                        dims=[-2, -1]
                    ).to(storage_device)
                _aug_counter += 1
                # rotation 180 deg
                # -> topologically the same as K1 + K2 flip
                #
                # rotation 270 deg
                logits_val[i:max_i] += torch.rot90(
                        wrs_fn(inference_model(
                            torch.rot90(images[i:max_i], k=3, dims=[-2, -1])
                        )),
                        k=1,
                        dims=[-2, -1]
                    ).to(storage_device)
                _aug_counter += 1
                # normal prediction
                logits_val[i:max_i] += wrs_fn(inference_model(images[i:max_i])).to(storage_device)
                _aug_counter += 1
                # median logits
                logits_val[i:max_i] /= _aug_counter
                '''
                # 1. Generate augmented copies and forward pass
                # Collect raw outputs from 5 distinct transformations:
                # H-flip, V-flip, 90° rotation, 270° rotation, and original.
                # Note: 180° rotation is omitted as it's topologically identical to H-flip + V-flip in 2D.
                aug_outputs = [
                    torch.flip(inference_model(torch.flip(images[i:max_i], dims=[-2])), dims=[-2]),      # H-flip
                    torch.flip(inference_model(torch.flip(images[i:max_i], dims=[-1])), dims=[-1]),      # V-flip
                    torch.rot90(inference_model(torch.rot90(images[i:max_i], k=1, dims=[-2, -1])), k=3, dims=[-2, -1]),  # 90° rot
                    torch.rot90(inference_model(torch.rot90(images[i:max_i], k=2, dims=[-2, -1])), k=2, dims=[-2, -1]),  # 180° rot
                    torch.rot90(inference_model(torch.rot90(images[i:max_i], k=3, dims=[-2, -1])), k=1, dims=[-2, -1]),  # 270° rot
                    inference_model(images[i:max_i])                                                    # Original
                ]
                # 2. Stabilize and average across augmentations
                # Convert to probabilities, clamp to [1e-7, 1-1e-7] to prevent downstream log/division instability,
                # then compute the ensemble mean. Probability averaging is mathematically required for valid TTA
                # because it preserves calibration and decision boundaries, unlike raw logit averaging.
                probs = [torch.clamp(activation(p), min=1e-7, max=1.0 - 1e-7) for p in aug_outputs]
                probs = torch.stack(probs, dim=0).mean(dim=0).to(storage_device)
                if isinstance(activation, torch.nn.Softmax):
                    # Convert back to logits for consistency with non-augmented path
                    logits_val[i:max_i] = torch.log(probs)  
                else:
                    # sigmoid or other unknown activation
                    # logit function is the mathematically correct inverse of sigmoid, 
                    # and for softmax we can use log since the probabilities are already normalized across classes. 
                    # For other activations, we assume they are monotonic and apply logit as a general stabilizer, 
                    # but this may not be mathematically correct.
                    logits_val[i:max_i] = torch.logit(probs, eps=1e-7)
            else:
                # 3D TTA
                x = images[i:max_i]
                D, H, W = 2, 3, 4  # Spatial dims for (N, C, D, H, W)
                # 7 flipped variants + original
                aug_x = [
                    torch.flip(x, dims=[D]),
                    torch.flip(x, dims=[H]),
                    torch.flip(x, dims=[W]),
                    torch.flip(x, dims=[D, H]),
                    torch.flip(x, dims=[D, W]),
                    torch.flip(x, dims=[H, W]),
                    torch.flip(x, dims=[D, H, W]),
                    x  # <-- Original always last
                ]
                # Forward pass all augmentations
                aug_outputs = [inference_model(ax) for ax in aug_x]
                # Invert spatial transforms to align outputs with original coordinate system
                aug_outputs = [
                    torch.flip(aug_outputs[0], dims=[D]),
                    torch.flip(aug_outputs[1], dims=[H]),
                    torch.flip(aug_outputs[2], dims=[W]),
                    torch.flip(aug_outputs[3], dims=[D, H]),
                    torch.flip(aug_outputs[4], dims=[D, W]),
                    torch.flip(aug_outputs[5], dims=[H, W]),
                    torch.flip(aug_outputs[6], dims=[D, H, W]),
                    aug_outputs[7]  # Original is already aligned
                ]
                probs = [torch.clamp(activation(p), min=1e-7, max=1.0 - 1e-7) for p in aug_outputs]
                probs = torch.stack(probs, dim=0).mean(dim=0).to(storage_device)
                if isinstance(activation, torch.nn.Softmax):
                    logits_val[i:max_i] = torch.log(probs)  
                else:
                    logits_val[i:max_i] = torch.logit(probs, eps=1e-7)

            # check for nans or infs in logits
            if torch.isnan(logits_val[i:max_i]).any() or torch.isinf(logits_val[i:max_i]).any():
                print("WARNING:[predict_minibatches()] model output logits contain nans or inf, this should not happen ")

        # save features inside the model if requested
        ### known issue ###
        # If augmentations are requested, the features saved will be only 
        # those of the non-augmented image
        # and not the median of the features across all augmentations. 
        # This is because the features are saved inside the model during 
        # the forward pass, and they are overwritten at each forward pass. 
        # To save also the features of all augmentations, we should modify 
        # the model to save the features in a list and then concatenate them at 
        # the end of the inference. However, this would require more memory and 
        # a more complex implementation, so for now we will only save the features 
        # of the last augmentation.
        for f in model_features_to_save:
            try:
                f_attr = getattr(model, f)
                if torch.is_tensor(f_attr) and torch.is_tensor(features[f]):
                    features[f][i:max_i] += f_attr.to(output_logits_dtype).to(storage_device)
                else:
                    features[f] = None
            except Exception as e:
                raise ValueError(f"Model has no attribute '{f}'\nThis exception was raised:\n{e}")
    
    if verbose:
        log(f"Done in {int(time.time()-_t0):d} seconds")
    
    return logits_val, features

class PaddingWrapper(torch.nn.Module):
    """
    Wrapper class used only and exclusively
    to pad the input of a model to ensure that 
    the spatial dimensions are multiples of some values, 
    and then crop the output back to the original size. 
    This is useful when the model has some downsampling layers 
    that require the input spatial dimensions to be multiples 
    of some values (e.g. 32 for ResNet-based models). 
    The padding is applied only on the right and bottom sides 
    of the image, and only if needed. The output is cropped back
    to the original size by slicing it according to the 
    original spatial dimensions of the input.
    """
    def __init__(self, model: torch.nn.Module, multiple_of: tuple[int]):
        super().__init__()
        self.model = model
        self.multiple_of = multiple_of
        if len(self.multiple_of) < 2:
            raise ValueError(f"multiple_of should be a tuple of length at least 2, got {self.multiple_of}.")

    def forward(self, x):
        spatial_shape = x.shape[2:]
        padding = []
        # Reversed order for torch.nn.functional.pad (last dim first)
        for i, size in enumerate(reversed(spatial_shape)):
            m = self.multiple_of[-(i+1)]
            if m > 0:
                pad_amt = (m - size % m) % m
                # pad is (left, right) -> we do (0, pad_amt) for simplicity
                padding.extend([0, pad_amt])
            else:
                padding.extend([0, 0])
        # apply
        x_padded = torch.nn.functional.pad(x, padding)
        # model forward
        out = self.model(x_padded)
        # crop back to original size
        slices = [slice(None), slice(None)] # B and C
        for i, size in enumerate(spatial_shape):
            slices.append(slice(0, size))
        out = out[tuple(slices)]
        return out


if __name__ == '__main__':
    pass
