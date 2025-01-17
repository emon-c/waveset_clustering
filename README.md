# waveset_clustering

This code combines wavesets with simple statistical learning models to put a potentially new spin on audio effects. A waveset (introduced in Trevor Wishart’s Audible Design) is defined as the segment of an audio signal between two consecutive upward zero-crossings. Waveset segmentation is the process of dividing an audio signal into its many wavesets. For simple waveforms, this divides the signal into pitch periods. For more complex audio signals (for example, signals containing noise or multiple simultaneous pitches), this segmentation is more unpredictable. As a result, waveset transformations tend to have a characteristic sound containing highly digital glitches and crackles. The intended goal of this experiment is to produce a unique kind of distortion for an input audio file.

The input parameters for the script are the following: 
-	C, clusters-per-second. This is derived from the number of wavesets per cluster and the length of the input signal 
-	W, the linear weighting factor of the k-means clustering algorithm
-	S, the spectral weighting factor of the k-means clustering algorithm

Modifying these three parameters alone is enough to produce a wide variety of sounds. The script is split into several processes which are separated within the code as different functions.
1.	Segmentation – the input audio is read and then split into its wavesets
2.	Feature computation – the feature vector Xi is computed for the i-th waveset. Three features are computed: length (the number of samples between the zero crossings) the RMS of the waveset’s samples, and the spectral centroids of the waveset’s samples
3.	Normalization – the three features are scaled so that they each have variance 1 
4.	Weighting – the samples are scaled by W, the weighting parameter to control how much the clustering stage emphasizes the differences in length and amplitude
5.	Clustering – k-means clustering is run on the feature vectors X, producing k clusters. For each cluster, one representative waveset (the one closest to the centroid) is selected
6.	Quantization – the original signal is recreated using the using the representative wavesets from each cluster

![readmegraphic](https://github.com/user-attachments/assets/6b02acd1-9976-44bb-bf03-9a9a42d7ce72)

The screenshot shows how the audio signal changes as the clusters-per-second parameter, the C value, is altered. As the clusters-per-second parameter reduces, the individual clusters become increasingly more apparent in the waveform. Reducing this parameter essentially turns an audio signal into a square wave, making vocal samples sound robotic.

This experiment simply attempts to produce uniquely distorted audio samples. The mutations and the harmonic deconstruction created with the combination of the waveset segmentation and the clustering algorithm produce a sound which cannot be replicated with any other method. This could be achieved in real time using a sliding k-means window / a “streaming” k-means algorithm. An interesting extension of this would be to filter the original signal to mess up the zero-crossings and route it back to itself with delay. More musical extensions of the present work could include training a Markov chain on the sequenced wavesets to produce a similar result. There are many concatenative synthesizers which contain machine-listening features, however the method presented here demonstrates that a simple segmentation algorithm can be combined with a “dumb” algorithm to produce a distinctive result.
