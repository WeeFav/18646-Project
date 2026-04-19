"""
Verify GPU rasterization output against CPU baseline.

Compares TGA images pixel-by-pixel from:
  - GPU output: utah_teapot_results/res_*/gpu_out_e*_l*.tga
  - CPU baseline: utah_teapot_results_baseline/res_*/out_e*_l*.tga

Reports per-pixel color differences with statistics.
"""

import sys
import os
from pathlib import Path
import numpy as np
import struct

def load_tga(path):
    """Load TGA file (handles RLE compression)."""
    try:
        with open(path, 'rb') as f:
            # Read TGA header
            f.seek(0)
            id_len = struct.unpack('B', f.read(1))[0]
            color_map_type = struct.unpack('B', f.read(1))[0]
            image_type = struct.unpack('B', f.read(1))[0]

            f.seek(12)  # Skip to width/height
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            bits_per_pixel = struct.unpack('B', f.read(1))[0]
            descriptor = struct.unpack('B', f.read(1))[0]

            # Skip ID field
            f.seek(18 + id_len)

            # Read pixel data
            bytes_per_pixel = bits_per_pixel // 8
            if bytes_per_pixel not in [3, 4]:
                raise ValueError(f"Unsupported bits per pixel: {bits_per_pixel}")

            pixels = []

            # Image type 2 = uncompressed, 10 = RLE compressed
            if image_type == 2:  # Uncompressed
                for _ in range(width * height):
                    pixel = f.read(bytes_per_pixel)
                    pixels.append(pixel)
            elif image_type == 10:  # RLE compressed
                pixel_count = 0
                while pixel_count < width * height:
                    header = struct.unpack('B', f.read(1))[0]
                    rle_count = (header & 0x7f) + 1

                    if header & 0x80:  # RLE packet
                        pixel = f.read(bytes_per_pixel)
                        for _ in range(rle_count):
                            pixels.append(pixel)
                    else:  # Raw packet
                        for _ in range(rle_count):
                            pixels.append(f.read(bytes_per_pixel))

                    pixel_count += rle_count
            else:
                raise ValueError(f"Unsupported image type: {image_type}")

            # Convert pixels to numpy array (convert BGR to RGB)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for i, pixel in enumerate(pixels[:width*height]):
                y = i // width
                x = i % width
                if bytes_per_pixel == 4:
                    b, g, r, a = pixel[0], pixel[1], pixel[2], pixel[3]
                else:
                    b, g, r = pixel[0], pixel[1], pixel[2]
                img[y, x] = [r, g, b]  # Convert BGR to RGB

            return img
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def compare_images(gpu_path, cpu_path, tolerance=2):
    """
    Compare GPU and CPU output images.

    Returns:
        dict with keys: match_count, total_pixels, max_error, mean_error,
                        match_percentage, per_channel_errors
    """
    gpu_img = load_tga(gpu_path)
    cpu_img = load_tga(cpu_path)

    if gpu_img is None or cpu_img is None:
        return None

    if gpu_img.shape != cpu_img.shape:
        print(f"  Shape mismatch: GPU {gpu_img.shape} vs CPU {cpu_img.shape}")
        return None

    # Compute per-pixel differences (L2 norm across RGB channels)
    diff = np.abs(gpu_img.astype(float) - cpu_img.astype(float))
    pixel_errors = np.linalg.norm(diff, axis=2, ord=2)  # sqrt(R^2 + G^2 + B^2)

    max_error = np.max(pixel_errors)
    mean_error = np.mean(pixel_errors)
    median_error = np.median(pixel_errors)
    match_count = np.sum(pixel_errors <= tolerance)
    total_pixels = pixel_errors.size
    match_percentage = 100.0 * match_count / total_pixels

    # Per-channel statistics
    channel_errors = {
        'R': np.mean(diff[:, :, 0]),
        'G': np.mean(diff[:, :, 1]),
        'B': np.mean(diff[:, :, 2]),
    }

    return {
        'match_count': match_count,
        'total_pixels': total_pixels,
        'max_error': max_error,
        'mean_error': mean_error,
        'median_error': median_error,
        'match_percentage': match_percentage,
        'per_channel_errors': channel_errors,
        'tolerance': tolerance,
    }

def main():
    base_dir = Path(r'C:\Users\kkhua\Desktop\CMU')
    # base_dir = Path('~/private/18646/18646-Project').expanduser()
    gpu_results = base_dir / 'utah_teapot_results_tile_32x16'
    cpu_results = base_dir / 'utah_teapot_results_baseline'

    if not gpu_results.exists():
        print(f"GPU results not found: {gpu_results}")
        return 1

    if not cpu_results.exists():
        print(f"CPU baseline not found: {cpu_results}")
        print("  (Run main.cpp to generate baseline)")
        return 1

    # Verify only selected resolutions.
    target_res_dirs = ['res_16']
    gpu_images = []
    for res_dir in target_res_dirs:
        gpu_images.extend(sorted((gpu_results / res_dir).glob('gpu_out_e*_l*.tga')))

    if not gpu_images:
        print(f"No GPU output images found in {gpu_results}")
        return 1

    print(f"Found {len(gpu_images)} GPU output images. Comparing with baseline...")
    print()

    all_results = []
    total_pass = 0
    total_fail = 0

    for gpu_img in gpu_images:
        # Construct corresponding CPU baseline path
        rel_path = gpu_img.relative_to(gpu_results)
        # Strip "gpu_" prefix from filename for CPU baseline
        filename = rel_path.name.replace('gpu_out_', 'cpu_out_')
        cpu_img = cpu_results / rel_path.parent / filename

        if not cpu_img.exists():
            print(f"✗ {rel_path}: baseline not found")
            total_fail += 1
            continue

        result = compare_images(gpu_img, cpu_img, tolerance=2)

        if result is None:
            print(f"✗ {rel_path}: comparison failed")
            total_fail += 1
            continue

        all_results.append((rel_path, result))

        # Determine pass/fail (allow some tolerance for floating point differences)
        if result['match_percentage'] >= 99.0:  # 99% of pixels within tolerance=2
            status = "✓ PASS"
            total_pass += 1
        else:
            status = "✗ FAIL"
            total_fail += 1

        print(f"{status} {rel_path}")
        print(f"     Match: {result['match_count']}/{result['total_pixels']} "
              f"({result['match_percentage']:.1f}%)")
        print(f"     Max error: {result['max_error']:.1f}, "
              f"Mean error: {result['mean_error']:.2f}, "
              f"Median error: {result['median_error']:.2f}")
        print(f"     Per-channel mean error: R={result['per_channel_errors']['R']:.2f}, "
              f"G={result['per_channel_errors']['G']:.2f}, "
              f"B={result['per_channel_errors']['B']:.2f}")
        print()

    # Summary
    print("=" * 70)
    print(f"Summary: {total_pass} PASS, {total_fail} FAIL out of {len(all_results)} configs")
    print("=" * 70)

    if total_fail == 0:
        print("✓ All comparisons passed!")
        return 0
    else:
        print(f"✗ {total_fail} comparison(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())