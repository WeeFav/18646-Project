import sys
import math

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def compare_lines(line1, line2, eps=1e-6):
    tokens1 = line1.strip().split()
    tokens2 = line2.strip().split()

    if len(tokens1) != len(tokens2):
        return False

    for t1, t2 in zip(tokens1, tokens2):
        if is_float(t1) and is_float(t2):
            if not math.isclose(float(t1), float(t2), rel_tol=eps, abs_tol=eps):
                return False
        else:
            if t1 != t2:
                return False

    return True


def compare_files(file1, file2, eps=1e-6):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        line_num = 1

        while True:
            line1 = f1.readline()
            line2 = f2.readline()

            if not line1 and not line2:
                print("Files are equivalent within tolerance")
                return

            if not line1 or not line2:
                print(f"Files differ at line {line_num}:")
                print(f"{file1}: {line1.strip()}")
                print(f"{file2}: {line2.strip()}")
                return

            if not compare_lines(line1, line2, eps):
                print(f"Files differ at line {line_num}:")
                print(f"{file1}: {line1.strip()}")
                print(f"{file2}: {line2.strip()}")
                return

            line_num += 1


if __name__ == "__main__":
    eps = 1e-6
    
    eye_settings = [[0, 1, 3], [-3, 1, 0], [3, 1, 0], [0, 4, 0], [2, 2, 2]]
    light_setting = [[0, 1, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0]]
    
    for eye in eye_settings:
        for light in light_setting:
            compare_files(f"utah_teapot_results/res_16/raster_data_e{str(eye[0])}{str(eye[1])}{str(eye[2])}_l{str(light[0])}{str(light[1])}{str(light[2])}.txt", f"utah_teapot_results_baseline/res_16/raster_data_e{str(eye[0])}{str(eye[1])}{str(eye[2])}_l{str(light[0])}{str(light[1])}{str(light[2])}.txt", eps)