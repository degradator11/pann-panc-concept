use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::vision::{ImageFeatureMode, ImageResizeMode};

pub const FEATURE_BLOCK_COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PancGenome {
    pub image_size: u32,
    pub feature_mode: ImageFeatureMode,
    pub resize_mode: ImageResizeMode,
    pub threshold: f64,
    pub jaccard_weight: f64,
    pub top_k: usize,
    #[serde(default = "default_active_blocks")]
    pub active_blocks: u32,
}

impl PancGenome {
    pub fn random(space: &PancGenomeSpace, rng: &mut impl Rng) -> Self {
        Self {
            image_size: pick(&space.image_sizes, rng),
            feature_mode: pick(&space.feature_modes, rng),
            resize_mode: pick(&space.resize_modes, rng),
            threshold: rng.gen_range(space.threshold_min..=space.threshold_max),
            jaccard_weight: rng.gen_range(0.0..=1.0),
            top_k: pick(&space.top_k_values, rng),
            active_blocks: random_block_mask(rng),
        }
    }

    pub fn crossover(self, other: Self, rng: &mut impl Rng) -> Self {
        Self {
            image_size: if rng.gen_bool(0.5) {
                self.image_size
            } else {
                other.image_size
            },
            feature_mode: if rng.gen_bool(0.5) {
                self.feature_mode
            } else {
                other.feature_mode
            },
            resize_mode: if rng.gen_bool(0.5) {
                self.resize_mode
            } else {
                other.resize_mode
            },
            threshold: if rng.gen_bool(0.5) {
                self.threshold
            } else {
                other.threshold
            },
            jaccard_weight: if rng.gen_bool(0.5) {
                self.jaccard_weight
            } else {
                other.jaccard_weight
            },
            top_k: if rng.gen_bool(0.5) {
                self.top_k
            } else {
                other.top_k
            },
            active_blocks: crossover_block_mask(self.active_blocks, other.active_blocks, rng),
        }
    }

    pub fn mutate(&mut self, space: &PancGenomeSpace, mutation_rate: f64, rng: &mut impl Rng) {
        let rate = mutation_rate.clamp(0.0, 1.0);
        if rng.gen_bool(rate) {
            self.image_size = pick(&space.image_sizes, rng);
        }
        if rng.gen_bool(rate) {
            self.feature_mode = pick(&space.feature_modes, rng);
        }
        if rng.gen_bool(rate) {
            self.resize_mode = pick(&space.resize_modes, rng);
        }
        if rng.gen_bool(rate) {
            self.top_k = pick(&space.top_k_values, rng);
        }
        if rng.gen_bool(rate) {
            let delta = rng.gen_range(-space.threshold_step..=space.threshold_step);
            self.threshold =
                (self.threshold + delta).clamp(space.threshold_min, space.threshold_max);
        }
        if rng.gen_bool(rate) {
            let delta = rng.gen_range(-0.25..=0.25);
            self.jaccard_weight = (self.jaccard_weight + delta).clamp(0.0, 1.0);
        }
        if rng.gen_bool(rate) {
            mutate_block_mask(&mut self.active_blocks, rng);
        }
        self.active_blocks = normalize_block_mask(self.active_blocks);
    }

    pub fn similarity_name(self) -> &'static str {
        if self.jaccard_weight <= 0.05 {
            "hamming"
        } else if self.jaccard_weight >= 0.95 {
            "jaccard"
        } else {
            "hamming_jaccard_blend"
        }
    }

    pub fn active_block_count(self) -> usize {
        normalize_block_mask(self.active_blocks).count_ones() as usize
    }
}

fn default_active_blocks() -> u32 {
    all_blocks_mask()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PancGenomeSpace {
    pub image_sizes: Vec<u32>,
    pub feature_modes: Vec<ImageFeatureMode>,
    pub resize_modes: Vec<ImageResizeMode>,
    pub top_k_values: Vec<usize>,
    pub threshold_min: f64,
    pub threshold_max: f64,
    pub threshold_step: f64,
}

impl PancGenomeSpace {
    pub fn normalized(mut self) -> Self {
        if self.image_sizes.is_empty() {
            self.image_sizes.push(64);
        }
        if self.feature_modes.is_empty() {
            self.feature_modes.push(ImageFeatureMode::RichTexture);
        }
        if self.resize_modes.is_empty() {
            self.resize_modes.push(ImageResizeMode::CenterCrop);
        }
        if self.top_k_values.is_empty() {
            self.top_k_values.extend([1, 3, 5, 7]);
        }
        self.threshold_min = self.threshold_min.clamp(0.0, 1.0);
        self.threshold_max = self.threshold_max.clamp(self.threshold_min, 1.0);
        self.threshold_step = self.threshold_step.clamp(0.01, 1.0);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PancBinaryEvaluation {
    pub accuracy: f64,
    pub correct: usize,
    pub total: usize,
    pub predictions: Vec<usize>,
    pub memory_bytes: usize,
}

pub fn evaluate_panc_binary(
    library_samples: &[Vec<f64>],
    library_labels: &[usize],
    query_samples: &[Vec<f64>],
    query_labels: &[usize],
    class_count: usize,
    genome: PancGenome,
) -> PancBinaryEvaluation {
    let predictions = predict_panc_binary_batch(
        library_samples,
        library_labels,
        query_samples,
        class_count,
        genome,
    );
    let correct = predictions
        .iter()
        .zip(query_labels)
        .filter(|(predicted, expected)| **predicted == **expected)
        .count();
    let total = query_labels.len();
    PancBinaryEvaluation {
        accuracy: if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        },
        correct,
        total,
        predictions,
        memory_bytes: packed_memory_bytes(library_samples, genome.threshold),
    }
}

pub fn predict_panc_binary_batch(
    library_samples: &[Vec<f64>],
    library_labels: &[usize],
    query_samples: &[Vec<f64>],
    class_count: usize,
    genome: PancGenome,
) -> Vec<usize> {
    let library = pack_samples(library_samples, genome.threshold);
    let queries = pack_samples(query_samples, genome.threshold);
    let mask = ComparisonMask::new(
        library
            .first()
            .or_else(|| queries.first())
            .map(|sample| sample.len)
            .unwrap_or(0),
        genome.active_blocks,
    );
    queries
        .iter()
        .map(|query| predict_one(&library, library_labels, query, class_count, genome, &mask))
        .collect()
}

fn predict_one(
    library: &[PackedBinaryVector],
    labels: &[usize],
    query: &PackedBinaryVector,
    class_count: usize,
    genome: PancGenome,
    mask: &ComparisonMask,
) -> usize {
    let mut top = Vec::<(f64, usize)>::new();
    let top_k = genome.top_k.max(1);
    for (reference, label) in library.iter().zip(labels) {
        let score = blended_similarity(query, reference, genome.jaccard_weight, mask);
        push_top_candidate(&mut top, (score, *label), top_k);
    }

    let mut score_by_class = vec![0.0; class_count.max(1)];
    let mut count_by_class = vec![0usize; class_count.max(1)];
    for (score, label) in top {
        if let Some(class_score) = score_by_class.get_mut(label) {
            *class_score += score;
        }
        if let Some(class_count) = count_by_class.get_mut(label) {
            *class_count += 1;
        }
    }

    (0..score_by_class.len())
        .max_by(|left, right| {
            score_by_class[*left]
                .total_cmp(&score_by_class[*right])
                .then_with(|| count_by_class[*left].cmp(&count_by_class[*right]))
                .then_with(|| right.cmp(left))
        })
        .unwrap_or(0)
}

fn push_top_candidate(top: &mut Vec<(f64, usize)>, candidate: (f64, usize), top_k: usize) {
    if top.len() < top_k {
        top.push(candidate);
        return;
    }

    let Some((worst_index, worst)) = top.iter().enumerate().min_by(|(_, left), (_, right)| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| right.1.cmp(&left.1))
    }) else {
        return;
    };
    let better = candidate
        .0
        .total_cmp(&worst.0)
        .then_with(|| worst.1.cmp(&candidate.1))
        .is_gt();
    if better {
        top[worst_index] = candidate;
    }
}

fn blended_similarity(
    left: &PackedBinaryVector,
    right: &PackedBinaryVector,
    jaccard_weight: f64,
    mask: &ComparisonMask,
) -> f64 {
    let jaccard_weight = jaccard_weight.clamp(0.0, 1.0);
    let hamming = hamming_similarity(left, right, mask);
    let jaccard = jaccard_similarity(left, right, mask);
    hamming * (1.0 - jaccard_weight) + jaccard * jaccard_weight
}

fn hamming_similarity(
    left: &PackedBinaryVector,
    right: &PackedBinaryVector,
    mask: &ComparisonMask,
) -> f64 {
    if mask.selected_bits == 0 {
        return 0.0;
    }

    let mismatches = left
        .words
        .iter()
        .zip(&right.words)
        .zip(&mask.word_masks)
        .map(|((left, right), mask)| ((left ^ right) & mask).count_ones() as usize)
        .sum::<usize>();
    (mask.selected_bits - mismatches.min(mask.selected_bits)) as f64 / mask.selected_bits as f64
}

fn jaccard_similarity(
    left: &PackedBinaryVector,
    right: &PackedBinaryVector,
    mask: &ComparisonMask,
) -> f64 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for ((left, right), mask) in left.words.iter().zip(&right.words).zip(&mask.word_masks) {
        intersection += (left & right & mask).count_ones() as usize;
        union += ((left | right) & mask).count_ones() as usize;
    }

    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn packed_memory_bytes(samples: &[Vec<f64>], threshold: f64) -> usize {
    pack_samples(samples, threshold)
        .iter()
        .map(|sample| sample.words.len() * std::mem::size_of::<u64>())
        .sum()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedBinaryVector {
    len: usize,
    words: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComparisonMask {
    word_masks: Vec<u64>,
    selected_bits: usize,
}

impl ComparisonMask {
    fn new(feature_len: usize, active_blocks: u32) -> Self {
        if feature_len == 0 {
            return Self {
                word_masks: Vec::new(),
                selected_bits: 0,
            };
        }

        let active_blocks = normalize_block_mask(active_blocks);
        let mut word_masks = vec![0_u64; feature_len.div_ceil(64)];
        for block in 0..FEATURE_BLOCK_COUNT {
            if (active_blocks & (1_u32 << block)) == 0 {
                continue;
            }
            let start = feature_len * block / FEATURE_BLOCK_COUNT;
            let end = feature_len * (block + 1) / FEATURE_BLOCK_COUNT;
            for bit in start..end {
                word_masks[bit / 64] |= 1_u64 << (bit % 64);
            }
        }

        let selected_bits = word_masks
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        Self {
            word_masks,
            selected_bits,
        }
    }
}

fn pack_samples(samples: &[Vec<f64>], threshold: f64) -> Vec<PackedBinaryVector> {
    samples
        .iter()
        .map(|sample| pack_sample(sample, threshold))
        .collect()
}

fn pack_sample(sample: &[f64], threshold: f64) -> PackedBinaryVector {
    let mut words = vec![0; sample.len().div_ceil(64)];
    for (index, value) in sample.iter().copied().enumerate() {
        if value >= threshold {
            words[index / 64] |= 1_u64 << (index % 64);
        }
    }
    PackedBinaryVector {
        len: sample.len(),
        words,
    }
}

fn pick<T: Copy>(values: &[T], rng: &mut impl Rng) -> T {
    values[rng.gen_range(0..values.len())]
}

pub fn all_blocks_mask() -> u32 {
    (1_u32 << FEATURE_BLOCK_COUNT) - 1
}

pub fn normalize_block_mask(mask: u32) -> u32 {
    let mask = mask & all_blocks_mask();
    if mask == 0 { all_blocks_mask() } else { mask }
}

pub fn format_block_mask(mask: u32) -> String {
    format!("0x{:04x}", normalize_block_mask(mask))
}

fn random_block_mask(rng: &mut impl Rng) -> u32 {
    if rng.gen_bool(0.25) {
        return all_blocks_mask();
    }

    let mut mask = 0_u32;
    for block in 0..FEATURE_BLOCK_COUNT {
        if rng.gen_bool(0.65) {
            mask |= 1_u32 << block;
        }
    }
    normalize_block_mask(mask)
}

fn crossover_block_mask(left: u32, right: u32, rng: &mut impl Rng) -> u32 {
    let mut mask = 0_u32;
    for block in 0..FEATURE_BLOCK_COUNT {
        let bit = 1_u32 << block;
        if rng.gen_bool(0.5) {
            mask |= left & bit;
        } else {
            mask |= right & bit;
        }
    }
    normalize_block_mask(mask)
}

fn mutate_block_mask(mask: &mut u32, rng: &mut impl Rng) {
    if rng.gen_bool(0.15) {
        *mask = all_blocks_mask();
        return;
    }

    let toggles = rng.gen_range(1..=3);
    for _ in 0..toggles {
        let block = rng.gen_range(0..FEATURE_BLOCK_COUNT);
        *mask ^= 1_u32 << block;
    }
    *mask = normalize_block_mask(*mask);
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn evolved_binary_panc_classifies_simple_vectors() {
        let library = vec![vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let labels = vec![0, 1];
        let queries = vec![vec![0.9, 0.8, 0.0], vec![0.0, 0.1, 0.9]];
        let query_labels = vec![0, 1];
        let genome = PancGenome {
            image_size: 4,
            feature_mode: ImageFeatureMode::Pixels,
            resize_mode: ImageResizeMode::Stretch,
            threshold: 0.5,
            jaccard_weight: 0.5,
            top_k: 1,
            active_blocks: all_blocks_mask(),
        };

        let evaluation =
            evaluate_panc_binary(&library, &labels, &queries, &query_labels, 2, genome);

        assert_eq!(evaluation.predictions, vec![0, 1]);
        assert_eq!(evaluation.accuracy, 1.0);
    }

    #[test]
    fn genome_mutation_stays_inside_search_space() {
        let space = PancGenomeSpace {
            image_sizes: vec![16, 32],
            feature_modes: vec![ImageFeatureMode::Pixels],
            resize_modes: vec![ImageResizeMode::Stretch],
            top_k_values: vec![1, 3],
            threshold_min: 0.2,
            threshold_max: 0.8,
            threshold_step: 0.3,
        }
        .normalized();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut genome = PancGenome::random(&space, &mut rng);

        for _ in 0..100 {
            genome.mutate(&space, 1.0, &mut rng);
            assert!(space.image_sizes.contains(&genome.image_size));
            assert!(space.feature_modes.contains(&genome.feature_mode));
            assert!(space.resize_modes.contains(&genome.resize_mode));
            assert!(space.top_k_values.contains(&genome.top_k));
            assert!((0.2..=0.8).contains(&genome.threshold));
            assert!((0.0..=1.0).contains(&genome.jaccard_weight));
            assert_ne!(genome.active_blocks, 0);
            assert_eq!(genome.active_blocks & !all_blocks_mask(), 0);
        }
    }

    #[test]
    fn block_mask_can_ignore_unhelpful_descriptor_region() {
        let mut class_zero = vec![0.0; FEATURE_BLOCK_COUNT];
        class_zero[0] = 1.0;
        class_zero[1] = 1.0;
        class_zero[14] = 0.0;
        class_zero[15] = 0.0;
        let mut class_one = vec![0.0; FEATURE_BLOCK_COUNT];
        class_one[0] = 0.0;
        class_one[1] = 0.0;
        class_one[14] = 1.0;
        class_one[15] = 1.0;
        let mut query = class_zero.clone();
        query[14] = 1.0;
        query[15] = 1.0;

        let library = vec![class_zero, class_one];
        let labels = vec![0, 1];
        let queries = vec![query];
        let query_labels = vec![0];
        let genome = PancGenome {
            image_size: 4,
            feature_mode: ImageFeatureMode::Pixels,
            resize_mode: ImageResizeMode::Stretch,
            threshold: 0.5,
            jaccard_weight: 0.0,
            top_k: 1,
            active_blocks: 0b0000_0000_0000_0011,
        };

        let evaluation =
            evaluate_panc_binary(&library, &labels, &queries, &query_labels, 2, genome);

        assert_eq!(evaluation.predictions, vec![0]);
        assert_eq!(evaluation.accuracy, 1.0);
    }
}
