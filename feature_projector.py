import os
import torch
import numpy as np
import argparse 
from autoencoder.model import Autoencoder

def decode_features(dataname):

    autoencoder_ckpt_path = f"./autoencoder/ckpt/{dataname}/best_ckpt.pth"
    feature_dir = f"./output/{dataname}/test/ours_30000_text/feature_map_npy"

    print(f"Dataset: {dataname}")
    print(f"Loading checkpoint from: {autoencoder_ckpt_path}")
    print(f"Loading features from: {feature_dir}")


    if not os.path.exists(autoencoder_ckpt_path):
        print(f"Error: Checkpoint file not found at -> {autoencoder_ckpt_path}")
        return

    if not os.path.isdir(feature_dir):
        print(f"Error: Feature directory not found at -> {feature_dir}")
        return

    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)

    checkpoint = torch.load(autoencoder_ckpt_path, map_location=device)
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()

    output_dir = os.path.join(feature_dir, "decoded")
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(feature_dir)):
        if filename.endswith(".npy"):
            feature_path = os.path.join(feature_dir, filename)      
            feature_map = np.load(feature_path)

            H, W = feature_map.shape[1], feature_map.shape[2]
            feature_reshaped = feature_map.reshape(3, -1).T  # Shape: (H * W, 3)
            feature_reshaped_tensor = torch.from_numpy(feature_reshaped).float().to(device)

            # Decode each 3-dimensional feature to 512 dimensions
            with torch.no_grad():
                feature_512_tensor = autoencoder.decode(feature_reshaped_tensor)

            # Reshape the result back to (H, W, 512)
            feature_512 = feature_512_tensor.cpu().numpy()
            feature_512_reshaped = feature_512.reshape(H, W, -1)

            # Save the decoded feature map
            output_path = os.path.join(output_dir, f"decoded_{filename}")
            np.save(output_path, feature_512_reshaped)
            print(f"Saved: {output_path}, Shape: {feature_512_reshaped.shape}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Decodes 3D feature maps to 512D using an Autoencoder.")
    parser.add_argument("dataname", type=str, help="The name of the dataset to process (e.g., 'lerf/teatime').")
    
    args = parser.parse_args()
    decode_features(args.dataname)