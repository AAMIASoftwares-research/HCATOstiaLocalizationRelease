"""
Converts one list of (image, ostium source) pairs into .ostium.json files.
Superseded (2026-10-05) by rebuild_ostia.py, which rebuilds OSTIAv2/JSON and OSTIAv2/Slicer for every
dataset from the sources, recomputes the ImageCAS 1-500 consensus from the annotators' files and
checks it. Kept for converting arbitrary lists; its _to_convert/ lists point to the source locations.
"""
import os
import sys
import argparse

from hcatostialocalization.io.io import extract_simplified_ras, compute_image_metadata, save_blueprint_json, read_image_header_lps

def get_base_filename(filename):
    """
    Removes common medical image extensions to get the base file name.
    Handles compound extensions like .nii.gz or .img.nii.gz.
    """
    base_name = filename
    # List extensions from longest to shortest to ensure proper removal
    extensions = ['.img.nii.gz', '.nii.gz', '.nii', '.mha', '.nrrd', '.mhd']
    
    for ext in extensions:
        if base_name.endswith(ext):
            return base_name[:-len(ext)]
            
    # Fallback to standard splitext if none of the above match
    return os.path.splitext(base_name)[0]

def process_dataset(images_txt, ostia_txt, output_dir, dataset_name):
    """
    Reads lists of images and ostia files, processes them, and outputs standardized JSONs.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read file paths
    try:
        with open(images_txt, 'r') as f_img, open(ostia_txt, 'r') as f_ost:
            image_paths = [line.strip() for line in f_img if line.strip()]
            ostia_paths = [line.strip() for line in f_ost if line.strip()]
    except IOError as e:
        print(f"Error reading input lists: {e}")
        return

    # Validate lengths
    if len(image_paths) != len(ostia_paths):
        print(f"Error: Mismatch in file list lengths. "
              f"Images: {len(image_paths)}, Ostia files: {len(ostia_paths)}")
        return

    print(f"Starting processing for {len(image_paths)} image-ostium pairs...\n")

    success_count = 0
    error_count = 0
    warning_ids = []

    for img_path, ost_path in zip(image_paths, ostia_paths):
        if not os.path.exists(img_path):
            print(f"[-] Image not found: {img_path}")
            error_count += 1
            continue
            
        if not os.path.exists(ost_path):
            print(f"[-] Ostium file not found: {ost_path}")
            error_count += 1
            continue

        filename = os.path.basename(img_path)
        base_name = get_base_filename(filename)
        
        # Format the Image ID as requested
        image_id = f"{dataset_name}__{base_name}"
        
        # Ensure filepath ends in .ostium.json as mandated by the IO module
        out_filepath = os.path.join(output_dir, f"{image_id}.ostium.json")
        
        print(f"Processing: {image_id}")
        
        # 1. Extract RAS Coordinates (Auto-detects format; the image header is needed for ASOCA)
        simple_pos = extract_simplified_ras(ost_path, image_path=img_path)
        if simple_pos is None or not simple_pos.get('R', {}).get('ras') or not simple_pos.get('L', {}).get('ras'):
            print(f"    [-] Failed to extract RAS coordinates (R and L) from {ost_path}")
            error_count += 1
            continue

        # 2. Open images and compute IJK & Affines
        try:
            full_pos, affine_ijk_to_ras, affine_ras_to_ijk = compute_image_metadata(img_path, simple_pos)
        except Exception as e:
            cause = e
            while cause.__cause__ is not None:  # MONAI wraps the real error in "applying transform ..."
                cause = cause.__cause__
            print(f"    [-] Error computing metadata for {img_path}: {type(cause).__name__}: {cause}")
            error_count += 1
            continue

        # 2b. Sanity check: both ostia must fall inside the image, otherwise the source
        #     coordinate frame was misread. The file is still written, but flagged.
        size = read_image_header_lps(img_path)[3]
        for side in ['R', 'L']:
            ijk = full_pos[side]['ijk']['sitk']
            if not all(-0.5 <= c <= s - 0.5 for c, s in zip(ijk, size)):
                warning_ids.append(f"{image_id} ({side})")
                print(f"    [!] WARNING: ostium {side} is outside the image: ijk={[round(c, 1) for c in ijk]}, size={size.tolist()}")

        # 3. Save the final Blueprint JSON
        comment = f"Automatically generated blueprint for {dataset_name}"
        result = save_blueprint_json(
            filepath=out_filepath,
            image_id=image_id,
            image_filename=filename,
            pos_dict=full_pos,
            affine_ijk_to_ras=affine_ijk_to_ras,
            affine_ras_to_ijk=affine_ras_to_ijk,
            comment=comment
        )
        
        if result:
            success_count += 1
        else:
            error_count += 1

    print("\n" + "="*40)
    print("Processing Complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Ostia outside their image: {len(warning_ids)}" + (f" -> {', '.join(warning_ids)}" if warning_ids else ""))
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image and ostium marker files into a standardized JSON dataset.")
    parser.add_argument("-i", "--images", required=True, help="Path to text file containing image file paths (one per line).")
    parser.add_argument("-o", "--ostia", required=True, help="Path to text file containing ostium JSON file paths (one per line).")
    parser.add_argument("-d", "--outdir", required=True, help="Directory to save the standardized .ostium.json files.")
    parser.add_argument("-n", "--dataset", required=True, help="Name of the dataset (used as a prefix for the image_id).")

    args = parser.parse_args()

    process_dataset(args.images, args.ostia, args.outdir, args.dataset)
