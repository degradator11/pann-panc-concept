use progress_ai::panc::{PancComparator, SimilarityMetric};
use progress_ai::pann::{Distributor, PannModel, one_hot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = vec![
        vec![0.1, 0.1],
        vec![0.2, 0.2],
        vec![0.8, 0.8],
        vec![0.9, 0.9],
    ];
    let labels = vec![0, 0, 1, 1];
    let targets = labels
        .iter()
        .map(|label| one_hot(*label, 2))
        .collect::<Vec<_>>();

    let mut pann = PannModel::with_unit_ranges(2, 4, 2, Distributor::HardBin)?;
    for _ in 0..3 {
        pann.train_epoch_difference(&samples, &targets)?;
    }

    println!(
        "PANN-like prototype training accuracy: {:.1}%",
        pann.accuracy(&samples, &labels)? * 100.0
    );

    let mut panc = PancComparator::new(SimilarityMetric::Euclidean);
    for (sample, label) in samples.iter().zip(&labels) {
        panc.add_reference(sample.clone(), *label, ())?;
    }

    let query = [0.85, 0.85];
    let prediction = panc.predict_label(&query, 3)?;
    let top = panc.top_k(&query, 2)?;
    println!("PANC-like comparator prediction for {query:?}: {prediction:?}");
    println!(
        "Top analogue labels: {:?}",
        top.iter().map(|n| n.label).collect::<Vec<_>>()
    );

    Ok(())
}
