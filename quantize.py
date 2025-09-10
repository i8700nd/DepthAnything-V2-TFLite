import tensorflow as tf
import numpy as np
import cv2
import os

SAVED_MODEL_DIR = "converted_tf_fp32"
REPRESENTATIVE_DATASET_DIR = "representative_images"
TFLITE_UINT8_MODEL_PATH = "depth_anything_v2_uint8.tflite"
MODEL_INPUT_SIZE = (518, 518)

def representative_dataset_gen():
    image_files = os.listdir(REPRESENTATIVE_DATASET_DIR)[:100]
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(REPRESENTATIVE_DATASET_DIR, image_name)
        img = cv2.imread(image_path)
        if img is None: continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, MODEL_INPUT_SIZE)

        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)
        print(f"  Calibrating with image #{i + 1}")
        yield [img]

print("--- Starting INT8 Quantization from SavedModel ---")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    print("\n--- Running converter. This will take a few minutes... ---")
    tflite_quant_model = converter.convert()

    with open(TFLITE_UINT8_MODEL_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"\n" + "=" * 50)
    print(f"--- SUCCESS! ---")
    print(f"UINT8 model saved to: {TFLITE_UINT8_MODEL_PATH}")
    print("=" * 50 + "\n")

except Exception as e:
    print(f"\n--- ERROR DURING CONVERSION ---")
    print(e)