import cv2
import mediapipe as mp
import csv
import os

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    csv_filename = 'asl_alphabet_dataset.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label', 'sample_id']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)
    
    # Define labels (A-Z + space + nothing)
    labels = [chr(i) for i in range(65, 91)]  # A-Z
    labels.extend(['space', 'nothing'])
    
    dataset_path = './dataset/asl_alphabet_train/'
    total_samples = 0
    
    print("Starting ASL dataset processing...")
    print(f"Processing folders: {labels}")
    
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        
        print(f"\nProcessing: {label}")
        
        # Get all .jpg files
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith('.jpg')]
        sample_count = 0
        
        for sample_id, image_file in enumerate(image_files, 1):
            image_path = os.path.join(label_path, image_file)
            
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                
                # Prepare row data
                row = [label, sample_id]
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                # Save to CSV
                with open(csv_filename, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
                
                sample_count += 1
                total_samples += 1
            
            # Print progress every 100 images
            if sample_id % 100 == 0:
                print(f"  Processed {sample_id} images...")
        
        print(f"  {label}: {sample_count} samples captured")
    
    hands.close()
    
    print(f"\nProcessing complete!")
    print(f"Total samples: {total_samples}")
    print(f"File saved: {csv_filename}")

if __name__ == "__main__":
    main()