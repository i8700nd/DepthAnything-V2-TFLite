import onnx2tf

ONNX_MODEL_PATH = "depth_anything_v2_vits.onnx"
OUTPUT_FOLDER_PATH = "depth_anything_v2_saved_model"

print(f"--- Starting ONNX to INT8 TFLite Conversion ---")
print(f"Input ONNX: {ONNX_MODEL_PATH}")
print(f"Output Folder: {OUTPUT_FOLDER_PATH}")

try:
    onnx2tf.convert(
        input_onnx_file_path=ONNX_MODEL_PATH,
        output_folder_path=OUTPUT_FOLDER_PATH,
        output_integer_quantized_tflite = True,
    )
    print(f"\n--- SUCCESS! ---")
    print(f"INT8 quantized models saved in: {OUTPUT_FOLDER_PATH}")

except Exception as e:
    print(f"\n--- ERROR DURING CONVERSION ---")
    print(e)