import coremltools as ct
import sys

def inspect_model(model_path):
    print(f"Inspecting model: {model_path}")
    try:
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        print("\n--- Inputs ---")
        for input_desc in spec.description.input:
            print(f"Name: {input_desc.name}")
            print(f"Type: {input_desc.type}")
            
        print("\n--- Outputs ---")
        for output_desc in spec.description.output:
            print(f"Name: {output_desc.name}")
            # print(f"Type: {output_desc.type}")
            # Try to get output shape if possible
            if hasattr(output_desc.type, 'multiArrayType'):
                print(f"Shape: {output_desc.type.multiArrayType.shape}")

        # Check for class labels in metadata or spec
        metadata = model.user_defined_metadata
        if 'class_labels' in metadata:
            print("\n--- Class Labels (Metadata) ---")
            print(metadata['class_labels'])
        else:
            # Check if it's a classifier spec
            if spec.WhichOneof('Type') == 'classifier':
                print("\n--- Class Labels (Classifier Spec) ---")
                labels = spec.classifierInterface.stringClassLabels.labels
                if not labels:
                    labels = spec.classifierInterface.int64ClassLabels.labels
                print(labels)
            else:
                # For non-classifier models, labels might be in a separate text file or not present
                print("\nClass labels not found in model spec.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_model("yolo26n.mlpackage")
