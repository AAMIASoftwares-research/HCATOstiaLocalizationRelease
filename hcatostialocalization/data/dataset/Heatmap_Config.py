class HeatmapParams:
    # Parameters for coronary ostia heatmap generation
    
    # Sigma in mm (will be converted to voxels based on spacing)
    # A sigma of 2.0 - 3.0 mm is usually a good starting point
    sigma = 2.0 
    
    # Maximum value of the Gaussian
    amplitude = 1.0
    
    # If True, generates two separate channels (one per ostium),
    # if False, generates a single channel with different labels
    multi_channel = False 
    
    # Identifiers in the label tensor
    labels_map = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium" 
    }

    # Background threshold: zeroes Gaussian values below this threshold
    # Useful to keep the mask sparse
    min_threshold = 0.01