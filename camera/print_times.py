import exifread
import os
from collections import defaultdict

def get_exposure_time(file_path):
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        aperture = tags['EXIF FNumber']
        aperture_val = aperture.values[0]
        assert float(aperture_val.num) / float(aperture_val.den) == 4.5, \
            f"{file_path}: expected aperture 4.5, got {float(aperture_val.num) / float(aperture_val.den)}"
        exposure = tags.get('EXIF ExposureTime')
        if exposure:
            val = exposure.values[0]
            return float(val.num) / float(val.den)
    return None

def analyze_folder(folder_path="."):
    # Dictionary to group filenames by exposure value
    # Key: exposure (float), Value: list of filenames
    groups = defaultdict(list)
    total_exposed_time = 0.0

    # 1. Collect and group NEF files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".nef"):
            path = os.path.join(folder_path, filename)
            exposure = get_exposure_time(path)
            if exposure is not None:
                groups[exposure].append(filename)
                total_exposed_time += exposure

    # 2. Sort the unique exposure values (keys)
    sorted_exposures = sorted(groups.keys())

    # 3. Print headers
    print(f"{'Exposure (s)':<15} | {'Count':<8} | {'Factor Increase'} | {'Sample File'}")
    print("-" * 70)

    # 4. Print grouped results
    for i, exposure in enumerate(sorted_exposures):
        count = len(groups[exposure])
        sample_file = groups[exposure][0] # Show the first file in this group
        
        if i == 0:
            factor_str = "(Baseline)"
        else:
            prev_exposure = sorted_exposures[i-1]
            factor = exposure / prev_exposure if prev_exposure > 0 else 0
            factor_str = f"x{factor:.4f}"

        print(f"{exposure:<15.6f} | {count:<8} | {factor_str:<15} | {sample_file}")

    # 5. Print Total Sum
    print("-" * 70)
    print(f"TOTAL EXPOSED TIME: {total_exposed_time:.6f} seconds")

if __name__ == "__main__":
    analyze_folder()
