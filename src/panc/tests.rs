use super::*;

#[test]
fn cosine_top_k_returns_best_analogue() {
    let mut comparator = PancComparator::new(SimilarityMetric::Cosine);
    comparator
        .add_reference(vec![1.0, 0.0], "left", "first")
        .unwrap();
    comparator
        .add_reference(vec![0.0, 1.0], "right", "second")
        .unwrap();

    let neighbors = comparator.top_k(&[0.9, 0.1], 1).unwrap();

    assert_eq!(neighbors[0].label, "left");
    assert_eq!(neighbors[0].metadata, "first");
}

#[test]
fn predict_label_uses_top_k_score_weighted_vote() {
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    comparator.add_reference(vec![0.0], "a", ()).unwrap();
    comparator.add_reference(vec![0.1], "b", ()).unwrap();
    comparator.add_reference(vec![0.2], "b", ()).unwrap();

    let prediction = comparator.predict_label(&[0.11], 3).unwrap();

    assert_eq!(prediction, Some("b"));
}

#[test]
fn hamming_similarity_thresholds_dense_vectors() {
    let mut comparator = PancComparator::new(SimilarityMetric::Hamming { threshold: 0.5 });
    comparator
        .add_reference(vec![1.0, 0.0, 1.0], "a", ())
        .unwrap();
    comparator
        .add_reference(vec![0.0, 1.0, 0.0], "b", ())
        .unwrap();

    let prediction = comparator.predict_label(&[0.7, 0.2, 0.9], 1).unwrap();

    assert_eq!(prediction, Some("a"));
}

#[test]
fn dense_jaccard_ranks_shared_active_features() {
    let mut comparator = PancComparator::new(SimilarityMetric::Jaccard { threshold: 0.5 });
    comparator
        .add_reference(vec![1.0, 0.0, 1.0], "a", ())
        .unwrap();
    comparator
        .add_reference(vec![1.0, 1.0, 0.0], "b", ())
        .unwrap();

    let prediction = comparator.predict_label(&[0.9, 0.1, 0.8], 1).unwrap();

    assert_eq!(prediction, Some("a"));
}

#[test]
fn explanation_reports_best_neighbor_and_ordered_features() {
    let mut comparator = PancComparator::new(SimilarityMetric::Euclidean);
    comparator
        .add_reference(vec![0.0, 1.0], "a", "meta")
        .unwrap();
    comparator
        .add_reference(vec![1.0, 0.0], "b", "other")
        .unwrap();

    let explanation = comparator.explain(&[0.1, 0.9], 1, 1).unwrap();

    assert_eq!(explanation.neighbors[0].label, "a");
    assert_eq!(explanation.contributions.len(), 1);
}

#[test]
fn binary_encoding_and_bit_similarities_work() {
    let encoder = BinaryEncoder::new(0.5);
    let left = encoder.encode(&[1.0, 0.0, 1.0, 0.0]);
    let right = encoder.encode(&[1.0, 1.0, 1.0, 0.0]);

    assert!(left.get(0));
    assert!(!left.get(1));
    assert_eq!(left.hamming_similarity(&right).unwrap(), 0.75);
    assert!((left.jaccard_similarity(&right).unwrap() - (2.0 / 3.0)).abs() < 1e-10);
}

#[test]
fn binary_comparator_ranks_and_explains() {
    let encoder = BinaryEncoder::new(0.5);
    let mut comparator = BinaryPancComparator::new(BinarySimilarityMetric::Jaccard);
    comparator
        .add_reference(encoder.encode(&[1.0, 0.0, 1.0]), "a", ())
        .unwrap();
    comparator
        .add_reference(encoder.encode(&[0.0, 1.0, 0.0]), "b", ())
        .unwrap();

    let query = encoder.encode(&[0.8, 0.1, 0.9]);
    let explanation = comparator.explain(&query, 1, 2).unwrap();

    assert_eq!(explanation.neighbors[0].label, "a");
    assert_eq!(explanation.contributions[0], (0, 1.0));
}
