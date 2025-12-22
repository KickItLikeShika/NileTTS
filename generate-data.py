import whisper
from pydub import AudioSegment
import os
import numpy as np
import pickle
from speechbrain.pretrained import EncoderClassifier

AUDIO_FILE = "path/to/audio.m4a"
OUTPUT_FOLDER = "path/to/output"
MODEL_SIZE = "large"
MAX_DURATION = 15
CENTROIDS_FILE = "speaker_centroids.pkl"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

def get_speaker_label(embedding, centroids):
    """Find most similar centroid using cosine similarity."""
    # normalize
    emb_norm = embedding / np.linalg.norm(embedding)
    cent_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # cosine similarity
    similarities = np.dot(cent_norm, emb_norm)
    return np.argmax(similarities)


def chunk_data():
    audio = AudioSegment.from_file(AUDIO_FILE)

    print(f"loading Whisper model ({MODEL_SIZE})")
    model = whisper.load_model(MODEL_SIZE)
    
    print(f"loading Speaker Encoder ({SPEAKER_MODEL_SOURCE})...")
    classifier = EncoderClassifier.from_hparams(source=SPEAKER_MODEL_SOURCE, savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    centroids = None
    if os.path.exists(CENTROIDS_FILE):
        with open(CENTROIDS_FILE, "rb") as f:
            centroids = pickle.load(f)
        print("Loaded speaker centroids.")
    else:
        raise FileNotFoundError(f"Speaker centroids file {CENTROIDS_FILE} not found")

    print("transcribing to find sentence boundaries")
    result = model.transcribe(AUDIO_FILE, language="ar")
    raw_segments = result['segments']

    grouped_chunks = []
    current_group = {
        "start": raw_segments[0]['start'],
        "end": raw_segments[0]['end'],
        "text": raw_segments[0]['text'],
        "segments": [raw_segments[0]]
    }

    for i in range(1, len(raw_segments)):
        seg = raw_segments[i]
        
        # calculate what the new duration WOULD be if we added this segment
        # duration = end of new segment - start of current group
        potential_duration = seg['end'] - current_group['start']
        
        if potential_duration <= MAX_DURATION:
            # add to current group
            current_group['end'] = seg['end']
            current_group['text'] += " " + seg['text'] # Combine text
            current_group['segments'].append(seg)
        else:
            # the group is full. save it to list and start a new group.
            grouped_chunks.append(current_group)
            current_group = {
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'],
                "segments": [seg]
            }
    
    # append the final group
    grouped_chunks.append(current_group)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    csv_path = os.path.join(OUTPUT_FOLDER, "metadata.csv")
    csv_exists = os.path.exists(csv_path)

    csv_lines = []
    if not csv_exists:
        csv_lines.append("audio_file|text|speaker_name")
    else:
        csv_lines = ["audio_file|text|speaker_name"]

    for i, chunk_data in enumerate(grouped_chunks):
        # pydub works in milliseconds
        start_ms = chunk_data['start'] * 1000
        end_ms = chunk_data['end'] * 1000
        
        # add a tiny bit of buffer (50ms) to ensure we don't clip the first/last letter
        # but ensure we don't go below 0
        start_ms = max(0, start_ms - 50) 
        end_ms = min(len(audio), end_ms + 50)

        chunk_audio = audio[start_ms:end_ms]
        
        base_name = f"chunk_{i:04d}"
        audio_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.wav")
        text_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
        
        chunk_audio.export(audio_path, format="wav")
        text_content = chunk_data['text'].strip()
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_content)

        # speaker ID
        speaker_name = "UNKNOWN"
        if centroids is not None:
            try:
                signal = classifier.load_audio(audio_path)
                emb = classifier.encode_batch(signal)
                emb_flat = emb.squeeze().numpy()
                label = get_speaker_label(emb_flat, centroids)
                speaker_name = f"SPEAKER_{label + 1:02d}"
            except Exception as e:
                print(f"error identifying speaker for {base_name}: {e}")
        
        rel_path = os.path.relpath(audio_path, start=os.getcwd())
        csv_lines.append(f"{rel_path}|{text_content}|{speaker_name}")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"created {len(grouped_chunks)} files and updated metadata in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    chunk_data()

