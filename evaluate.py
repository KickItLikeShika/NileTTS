import os
import sys
import time
import json
import argparse
import random
import gc
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from jiwer import wer, cer
import whisper
from speechbrain.pretrained import EncoderClassifier

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS

    
EVAL_CSV = "data/xtts_dataset/metadata_eval.csv"
DATA_ROOT = "data/xtts_dataset"
OUTPUT_BASE = "evaluation_results"

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_baseline_model(device):
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    return tts, "api"


def load_finetuned_model(checkpoint_path, config_path, vocab_path, device):
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=checkpoint_path, vocab_path=vocab_path, use_deepspeed=False)
    model.to(device)
    model.eval()
    return model, "direct"


def compute_speaker_similarity(encoder, wav1_path, wav2_path):
    """Compute cosine similarity between speaker embeddings."""
    emb1 = encoder.encode_batch(encoder.load_audio(wav1_path)).squeeze().cpu().numpy()
    emb2 = encoder.encode_batch(encoder.load_audio(wav2_path)).squeeze().cpu().numpy()
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)


def synthesize_audio(model, model_type, text, ref_audio_path, output_path, device):
    """Generate audio using the model."""
    start_time = time.time()
    
    if model_type == "api":
        model.tts_to_file(
            text=text,
            speaker_wav=ref_audio_path,
            language="ar",
            file_path=output_path
        )
    else:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=model.config.gpt_cond_len,
            max_ref_length=model.config.max_ref_len,
            sound_norm_refs=model.config.sound_norm_refs,
        )
        
        out = model.inference(
            text=text,
            language="ar",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        torchaudio.save(output_path, wav, 24000)
    synthesis_time = time.time() - start_time
    return synthesis_time


def load_eval_samples(eval_csv, data_root, num_samples, seed=42):
    df = pd.read_csv(eval_csv, sep="|")
    
    samples = []
    for _, row in df.iterrows():
        audio_path = os.path.join(data_root, row["audio_file"])
        if os.path.exists(audio_path):
            samples.append({
                "text": row["text"],
                "audio_file": audio_path,
                "speaker_name": row.get("speaker_name", "SPEAKER_01")
            })
    
    random.seed(seed)
    return random.sample(samples, min(num_samples, len(samples)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate XTTS model")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (e.g., baseline, finetuned_v1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to model config.json")
    parser.add_argument("--vocab", type=str, default="checkpoints/XTTS_v2.0_original_model_files/vocab.json", help="Path to vocab.json")
    parser.add_argument("--eval_csv", type=str, default=EVAL_CSV, help="Path to evaluation CSV")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--whisper_model", type=str, default="large", help="Whisper model size")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    
    output_dir = os.path.join(OUTPUT_BASE, args.exp_name)
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    
    samples = load_eval_samples(args.eval_csv, DATA_ROOT, args.num_samples, args.seed)
    print(f"eval: {len(samples)} samples.")
    
    sample_info = []
    for idx, sample in enumerate(samples):
        sample_info.append({
            "idx": idx,
            "text": sample["text"],
            "ref_audio": sample["audio_file"],
            "gen_audio": os.path.join(gen_dir, f"sample_{idx:04d}.wav"),
            "speaker": sample["speaker_name"]
        })
    
    print("loading tts model")
    if args.checkpoint and args.config:
        print(f"finetuned model from: {args.checkpoint}")
        model, model_type = load_finetuned_model(args.checkpoint, args.config, args.vocab, device)
    else:
        print("pretrained XTTS v2 (baseline)")
        model, model_type = load_baseline_model(device)
    print("model loaded")
    
    print(f"generating {len(samples)} audio samples")
    rtf_scores = []
    for idx, sample in enumerate(tqdm(samples, desc="generating audios")):
        gen_audio = sample_info[idx]["gen_audio"]
        try:
            synthesis_time = synthesize_audio(model, model_type, sample["text"], sample["audio_file"], gen_audio, device)
            wav, sr = torchaudio.load(gen_audio)
            audio_duration = wav.shape[1] / sr
            rtf = synthesis_time / audio_duration if audio_duration > 0 else 0
            rtf_scores.append(rtf)
        except Exception as e:
            print(f"\n  Error generating sample {idx}: {e}")
            rtf_scores.append(None)
    
    del model
    clear_gpu_memory()
    
    print(f"transcribing with whisper ({args.whisper_model})")
    whisper_model = whisper.load_model(args.whisper_model)
    
    transcriptions = []
    for idx, info in enumerate(tqdm(sample_info, desc="transcribing")):
        gen_audio = info["gen_audio"]
        if not os.path.exists(gen_audio):
            transcriptions.append("")
            continue
        try:
            result = whisper_model.transcribe(gen_audio, language="ar")
            transcriptions.append(result["text"].strip())
        except Exception as e:
            print(f"error transcribing sample {idx}: {e}")
            transcriptions.append("")
    
    del whisper_model
    clear_gpu_memory()
    
    print("computing speaker similarity")
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    similarity_scores = []
    for idx, info in enumerate(tqdm(sample_info, desc="Speaker similarity")):
        gen_audio = info["gen_audio"]
        ref_audio = info["ref_audio"]
        if not os.path.exists(gen_audio):
            similarity_scores.append(None)
            continue
        similarity = compute_speaker_similarity(speaker_encoder, ref_audio, gen_audio)
        similarity_scores.append(similarity)
    
    del speaker_encoder
    clear_gpu_memory()
    
    print("computing metrics")
    wer_scores = []
    cer_scores = []
    results_samples = []
    
    for idx, info in enumerate(sample_info):
        text = info["text"]
        transcription = transcriptions[idx]
        
        if not transcription or not os.path.exists(info["gen_audio"]):
            continue
        
        wer_score = wer(text, transcription)
        cer_score = cer(text, transcription)
        wer_scores.append(wer_score)
        cer_scores.append(cer_score)

        results_samples.append({
            "idx": idx,
            "text": text,
            "transcription": transcription,
            "wer": wer_score,
            "cer": cer_score,
            "speaker_similarity": similarity_scores[idx],
            "rtf": rtf_scores[idx],
            "ref_audio": info["ref_audio"],
            "gen_audio": info["gen_audio"],
            "speaker": info["speaker"]
        })
    
    valid_similarity = [s for s in similarity_scores if s is not None]
    valid_rtf = [r for r in rtf_scores if r is not None and r > 0]

    results = {
        "exp_name": args.exp_name,
        "checkpoint": args.checkpoint,
        "num_samples": len(wer_scores),
        "samples": results_samples,
        "metrics": {
            "wer_mean": float(np.mean(wer_scores)) if wer_scores else None,
            "cer_mean": float(np.mean(cer_scores)) if cer_scores else None,
            "speaker_similarity_mean": float(np.mean(valid_similarity)) if valid_similarity else None,
            "rtf_mean": float(np.mean(valid_rtf)) if valid_rtf else None,
        }
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"evaluation results: {args.exp_name}")
    print(f"samples evaluated: {results['num_samples']}")
    m = results["metrics"]
    print(f"metrics:")
    if m["wer_mean"] is not None:
        print(f"wer: {m['wer_mean']*100:.2f}%")
    if m["cer_mean"] is not None:
        print(f"cer: {m['cer_mean']*100:.2f}%")
    if m["speaker_similarity_mean"] is not None:
        print(f"speaker similarity: {m['speaker_similarity_mean']:.4f}")
    if m["rtf_mean"] is not None:
        print(f"rtf: {m['rtf_mean']:.3f}x")
    print(f"results saved to: {results_path}")
    print(f"generated audio: {gen_dir}/")


if __name__ == "__main__":
    main()
