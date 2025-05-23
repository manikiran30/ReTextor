from text_detection import detect_and_extract_text
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Processing image: {image_path}")
    
    detected_text = detect_and_extract_text(image_path)
    
    if detected_text:
        print("\nDetected Text:")
        print("-" * 50)
        print(detected_text)
        print("-" * 50)
    else:
        print("No text was detected in the image.")

if __name__ == "__main__":
    main()
