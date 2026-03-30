scale = 0.3

input_file = "utah_teapot.obj"
output_file = "utah_teapot_scaled.obj"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        if line.startswith("v "):
            parts = line.split()
            x = float(parts[1]) * scale
            y = float(parts[2]) * scale
            z = float(parts[3]) * scale
            f_out.write(f"v {x} {y} {z}\n")
        else:
            f_out.write(line)