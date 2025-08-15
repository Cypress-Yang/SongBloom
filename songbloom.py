import os
import argparse
import json
import torch
import torchaudio
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download
import toml  # Using the toml library for the new config format

# It's good practice to keep environment variable settings at the top
os.environ['DISABLE_FLASH_ATTN'] = "1"

from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler

def hf_download(repo_id="CypressYang/SongBloom", model_name="songbloom_full_150s", local_dir="./cache", **kwargs):
    """
    Downloads model and configuration files from Hugging Face Hub.
    Prints status messages for a better user experience.
    """
    print("Downloading necessary files from Hugging Face Hub...")
    
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.yaml", local_dir=local_dir, **kwargs
    )
    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_name}.pt", local_dir=local_dir, **kwargs
    )
    vae_cfg_path = hf_hub_download(
        repo_id=repo_id, filename="stable_audio_1920_vae.json", local_dir=local_dir, **kwargs
    )
    vae_ckpt_path = hf_hub_download(
        repo_id=repo_id, filename="autoencoder_music_dsp1920.ckpt", local_dir=local_dir, **kwargs
    )
    g2p_path = hf_hub_download(
        repo_id=repo_id, filename="vocab_g2p.yaml", local_dir=local_dir, **kwargs
    )
    
    print("All files downloaded successfully.")
    return cfg_path

def load_model_config(cfg_file, parent_dir="./") -> DictConfig:
    """
    Loads and resolves the OmegaConf configuration.
    """
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("concat", lambda *x: [item for sublist in x for item in sublist])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", OmegaConf.load)
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))
    
    return OmegaConf.load(cfg_file)

def main():
    """
    Main function to drive the song generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate songs with SongBloom using a user-friendly .songbloom configuration file."
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to your .songbloom configuration file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        help="Directory to save the generated audio files."
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        default="CypressYang/SongBloom", 
        help="Hugging Face repository ID for the model."
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="songbloom_full_150s", 
        help="The name of the model to use."
    )
    parser.add_argument(
        "--local-dir", 
        type=str, 
        default="./cache", 
        help="Local directory to cache downloaded models."
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default='float32', 
        choices=['float32', 'bfloat16'], 
        help="Data type for model inference."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'")
        return

    # --- Model Loading ---
    print("--- Initializing SongBloom ---")
    cfg_path = hf_download(args.repo_id, args.model_name, args.local_dir)
    cfg = load_model_config(cfg_path, parent_dir=args.local_dir)
  
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    print(f"Loading model with {args.dtype} precision...")
    model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype)
    model.set_generation_params(**cfg.inference)
    print("Model loaded successfully.")
          
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Song Generation ---
    print(f"\n--- Loading songs from {args.input_file} ---")
    song_requests = toml.load(args.input_file)
    
    for song_name, details in song_requests.items():
        print(f"\nProcessing song: '{song_name}'")
        
        lyrics = details.get("lyrics")
        prompt_wav_path = details.get("prompt_wav")
        output_name = details.get("output_name", song_name) # Default to song name if not provided
        n_samples = details.get("n_samples", 1)

        if not lyrics or not prompt_wav_path:
            print(f"  -> Skipping '{song_name}' due to missing 'lyrics' or 'prompt_wav'.")
            continue

        try:
            prompt_wav, sr = torchaudio.load(prompt_wav_path)
        except FileNotFoundError:
            print(f"  -> Skipping '{song_name}': Prompt audio not found at '{prompt_wav_path}'")
            continue
            
        if sr != model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
            
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype)
        prompt_wav = prompt_wav[..., :10 * model.sample_rate]
        
        for i in range(n_samples):
            print(f"  -> Generating sample {i + 1} of {n_samples}...")
            wav = model.generate(lyrics, prompt_wav)
            
            output_filename = f'{args.output_dir}/{output_name}_sample{i + 1}.flac'
            torchaudio.save(output_filename, wav[0].cpu().float(), model.sample_rate)
            print(f"  -> Saved to {output_filename}")

    print("\n--- All songs processed. ---")

if __name__ == "__main__":
    main()