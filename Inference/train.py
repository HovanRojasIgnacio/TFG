import os
from ultralytics import YOLO

def main():
    # 1. Initialize the YOLO26 Model Architecture
    # We choose 'yolo26s.pt' (Small variant) as it is the ideal benchmark choice for a TFG.
    # It outperforms older models (like YOLOv8n/s) while remaining light enough to train 
    # efficiently on a free cloud tier (like Google Colab T4) or local consumer GPUs.
    model = YOLO("yolo26s.pt")

    # 2. Execute Advanced Deep Learning Fine-Tuning Pipeline
    print("Starting YOLO26 Deep Learning Training Pipeline...")
    results = model.train(
        # --- Core Path Parameters ---
        data="yolo_dataset/dataset.yaml",   # Path to the data dictionary generated previously
        epochs=100,                         # 100 epochs allows the MuSGD optimizer to stabilize and converge
        imgsz=640,                          # Core input frame scaling dimension (matches COCO pretraining)
        batch=16,                           # Training batch size. Scale down to 8 if running out of VRAM (OOM error)
        device=0,                           # Targets CUDA GPU device 0. Set to "cpu" if local hardware lacks a GPU
        workers=4,                          # Parallel CPU threads assigned to load data images into memory

        # --- YOLO26 Native Technical Paradigm ---
        optimizer="MuSGD",                  # Activates the hybrid SGD + Muon orthogonalization optimizer
        end2end=True,                       # Ensures NMS-Free execution paths for zero post-processing latency

        # --- Academic Evaluation & Checkpoint Control ---
        save=True,                          # Periodically checkpoints weights to disk
        project="TFG_PPE_Detection",        # Groups all logs and charts inside a clean, master thesis directory
        name="YOLO26s_Base_Run",            # Specific sub-folder labeling this training iteration
        patience=15,                        # Early stopping mechanism. Aborts execution if mAP50 flatlines for 15 epochs
        plots=True,                         # Instructs python to generate graphs, matrices, and PR curves

        # --- Specialized Spatial Data Augmentations ---
        mosaic=0.5,                         # Blends multiple images into tiled frames. Set to 0.5 to match moderate datasets
        close_mosaic=10,                    # CRITICAL: Shuts down heavy mosaic distortion during the final 10 epochs
        scale=0.7,                          # Image scaling bounds to force model variance for distant PPE objects
        fliplr=0.5,                         # Horizontal flip probability to eliminate camera viewpoint bias
        hsv_h=0.015,                        # Minor shifts in color hue to simulate varying site conditions
        hsv_s=0.7,                          # Adjusts saturation levels to simulate bright sunlight vs. overcast weather
        hsv_v=0.4                           # Modifies exposure values to train the model to see in poorly lit conditions
    )

    print("\nTraining Phase Successfully Completed!")
    print(f"Optimal validation weights are stored at: {os.path.abspath('TFG_PPE_Detection/YOLO26s_Base_Run/weights/best.pt')}")

if __name__ == "__main__":
    main()