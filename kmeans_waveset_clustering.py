import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import soundfile as sf

def load_audio(file_path, sr=None):
    try:
        audio, sr = librosa.load(file_path, sr=sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def segment_wavesets(audio, sr, min_waveset_length=0.01):
    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    wavesets = []
    start = 0
    for end in zero_crossings:
        if (end - start) / sr >= min_waveset_length:
            waveset = audio[start:end]
            if len(waveset) > 0:
                wavesets.append(waveset)
            start = end
    return wavesets

def compute_features(wavesets, sr):
    lengths = np.array([len(waveset) for waveset in wavesets])
    rms = np.array([np.sqrt(np.mean(waveset**2)) if len(waveset) > 0 else 0 for waveset in wavesets])
    spectral_centroids = np.array([librosa.feature.spectral_centroid(y=waveset, sr=sr).mean() if len(waveset) > 0 else 0 for waveset in wavesets])
    return lengths, rms, spectral_centroids

def normalize_and_weight_features(lengths, rms, spectral_centroids, length_weight=3.0, rms_weight=1.0, spectral_weight=2.0):
    features = np.column_stack((lengths, rms, spectral_centroids))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features[:, 0] *= length_weight  # Scale length
    features[:, 1] *= rms_weight     # Scale RMS
    features[:, 2] *= spectral_weight  # Scale spectral centroid
    return features

def cluster_wavesets(features, sr, clusters_per_second, wavesets, audio):
    duration_seconds = len(audio) / sr

    k = max(1, int(clusters_per_second * duration_seconds))

    # Run K-means clustering with the correctly computed k
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(features)

    # Select a representative waveset for each cluster
    representatives = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_wavesets = [wavesets[idx] for idx in cluster_indices]
        
        max_len = max(len(ws) for ws in cluster_wavesets)
        
        # Pad each waveset to the max length and average them to form a composite representative
        padded_wavesets = [np.pad(ws, (0, max_len - len(ws)), mode='constant') for ws in cluster_wavesets]
        composite_representative = np.mean(padded_wavesets, axis=0)
        
        representatives.append(composite_representative)
    
    return representatives, labels


def quantize_audio(audio, wavesets, representatives, labels):
    quantized_audio = np.zeros_like(audio)
    position = 0
    for i, waveset in enumerate(wavesets):
        representative = representatives[labels[i]]
        waveset_length = len(waveset)
        rep_length = len(representative)

        if rep_length < waveset_length:
            padded_representative = np.pad(representative, (0, waveset_length - rep_length), 'constant')
        else:
            padded_representative = representative[:waveset_length]
        
        quantized_audio[position:position + waveset_length] = padded_representative
        position += waveset_length

    return quantized_audio

def process(file_path, output_path, clusters_per_second=2, length_weight=3.0, rms_weight=1.0, spectral_weight=2.0, sr=None):
    audio, sr = load_audio(file_path, sr=sr)
    
    
    wavesets = segment_wavesets(audio, sr)
    lengths, rms, spectral_centroids = compute_features(wavesets, sr)
    features = normalize_and_weight_features(lengths, rms, spectral_centroids, length_weight, rms_weight, spectral_weight)
    representatives, labels = cluster_wavesets(features, sr, clusters_per_second, wavesets, audio)
    quantized_audio = quantize_audio(audio, wavesets, representatives, labels)
    sf.write(output_path, quantized_audio, sr)
    print(f"Quantized audio saved to {output_path}")


    
process('input.wav', 'output.wav', clusters_per_second=20, length_weight=5.0, rms_weight=1.0, spectral_weight=1.0)
