def convert_to_c_array(tflite_model_path, output_file):
    with open('C:/Users/sheet/Desktop/kneeRevive/KneeRevive/model.tflite', "rb") as f:
        data = f.read()

    with open(output_file, "w") as out:
        out.write("const unsigned char model_tflite[] = {\n")
        for i, byte in enumerate(data):
            if i % 12 == 0:
                out.write("  ")
            out.write(f"0x{byte:02x}, ")
            if i % 12 == 11:
                out.write("\n")
        out.write("\n};\n")
        out.write(f"const unsigned int model_tflite_len = {len(data)};\n")

# Example usage:
convert_to_c_array("model.tflite", "model_data.cc")
