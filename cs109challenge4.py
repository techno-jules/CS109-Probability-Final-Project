import numpy as np
from mido import MidiFile
from collections import Counter
from math import log
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# Feature extraction from MIDI files
def extract_features(midi_file):
    pitches = []
    durations = []
    try:
        for msg in MidiFile(midi_file):
            if msg.type == 'note_on' and msg.velocity > 0:
                pitches.append(msg.note)
                durations.append(msg.time)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
    return pitches, durations

# Estimate probability distribution
def estimate_distribution(features):
    counts = Counter(features)
    total = sum(counts.values())
    if total == 0:  # Safeguard against empty data
        return {k: 1e-6 for k in range(128)}  # Assume a uniform small probability
    return {k: v / total for k, v in counts.items()}

# Calculate log-likelihood
def log_likelihood(data, dist):
    small_prob = 1e-6  # Smallest allowable probability
    likelihood = 0
    for x in data:
        prob = dist.get(x, small_prob)
        if prob <= 0:  # Avoid log(0)
            prob = small_prob
        likelihood += log(prob)
    return likelihood

# Classify a melody segment
def classify_segment(segment, dist_beethoven, dist_chopin):
    ll_beethoven = log_likelihood(segment, dist_beethoven)
    ll_chopin = log_likelihood(segment, dist_chopin)
    return 'Beethoven' if ll_beethoven > ll_chopin else 'Chopin'

# Combined density and probability distribution visualization
def plot_density_and_probability(features, dist, name):
    # Prepare data
    counts = Counter(features)
    data = list(counts.values())
    pitches = list(range(128))
    probs = [dist.get(p, 0) for p in pitches]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot density histogram with fits
    ax1 = axes[0]
    ax1.hist(features, bins=30, density=True, alpha=0.6, color='gray', label='Data')
    
    # Fit and plot normal distribution
    mu, std = norm.fit(features)
    x = np.linspace(min(features), max(features), 100)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'k', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')

    # Fit and plot binomial approximation
    n = max(features)
    p_binom = sum(features) / (len(features) * n)
    binom_dist = binom.pmf(range(n + 1), n, p_binom)
    ax1.plot(range(n + 1), binom_dist, 'r-', label=f'Binomial Approx. (n={n}, p={p_binom:.2f})')

    ax1.set_title(f'Density Analysis: {name}')
    ax1.set_xlabel('Pitch')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Plot probability distribution as bar chart
    ax2 = axes[1]
    ax2.bar(pitches, probs, color='blue', alpha=0.6)
    ax2.set_title(f'Probability Distribution: {name}')
    ax2.set_xlabel('MIDI Pitch')
    ax2.set_ylabel('Probability')

  
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load MIDI files and extract features
    pitches_beethoven, _ = extract_features('pieces/beethovensonata.mid')
    pitches_chopin, _ = extract_features('pieces/chopinetude.mid')

    # Estimate distributions for Beethoven and Chopin
    dist_beethoven = estimate_distribution(pitches_beethoven)
    dist_chopin = estimate_distribution(pitches_chopin)

    # Plot density and probability side by side
    plot_density_and_probability(pitches_beethoven, dist_beethoven, "Beethoven")
    plot_density_and_probability(pitches_chopin, dist_chopin, "Chopin")

    # Test melody segments
    test_segments = [
        [60, 62, 64, 65, 67],  # Middle segment
        [72, 74, 76, 77, 79],  # Higher-pitched segment
        [50, 52, 53, 55, 57],  # Lower-pitched segment
        [30, 40, 45, 43, 32],   # Very low pitch segment
        [90, 100, 95, 99, 101],   # Very high pitch segment
    ]

    for i, segment in enumerate(test_segments):
        result = classify_segment(segment, dist_beethoven, dist_chopin)
        print(f"Segment {i+1} ({segment}) is classified as: {result}")
