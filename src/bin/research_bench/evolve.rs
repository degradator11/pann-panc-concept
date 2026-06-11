use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use progress_ai::evolution::{
    EvolutionConfig, PancBinaryEvaluation, PancGenome, PancGenomeSpace, ScoredGenome,
    evaluate_panc_binary, evaluate_population_parallel,
};
use progress_ai::preprocess::{Dataset, min_max_ranges, min_max_scale, train_test_split_indices};
use progress_ai::vision::{
    ImageFeatureMode, ImageResizeMode, ImageVectorConfig, load_image_folder,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use super::{
    Args, CommandOutput, EvolutionGenerationRow, EvolutionReport, EvolvedPancGenomeReport,
    classification_metrics, required_data_path, write_evolution_history_csv,
};

pub fn run_evolve_panc_image_folder(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    let data_path = required_data_path(args)?;
    let space = genome_space(args);
    let config = EvolutionConfig {
        population_size: args.evolution_population,
        generations: args.evolution_generations,
        elite_count: args.evolution_elite_count,
        mutation_rate: args.evolution_mutation_rate,
        seed: args.seed,
        threads: args.evolution_threads,
    }
    .normalized();
    let variants = load_variants(data_path, args.eval_data_path.as_deref(), &space, args)?;
    let class_names = variants
        .first()
        .map(|variant| variant.class_names.clone())
        .ok_or("evolution search space produced no image variants")?;

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut population = (0..config.population_size)
        .map(|_| PancGenome::random(&space, &mut rng))
        .collect::<Vec<_>>();

    let mut rows = Vec::new();
    let mut final_scored = Vec::new();
    for generation in 0..config.generations {
        let mut scored = score_population(&population, config.threads, &variants, args);
        sort_scored(&mut scored);
        let best = scored
            .first()
            .ok_or("evolution produced an empty scored population")?;
        rows.push(evaluate_generation_row(generation, best, &variants)?);

        if generation + 1 == config.generations {
            final_scored = scored;
            break;
        }

        population = next_generation(&scored, &space, config, &mut rng);
    }

    sort_scored(&mut final_scored);
    let best = final_scored
        .first()
        .ok_or("evolution did not produce a final best genome")?;
    let best_variant = variant_for(&best.genome, &variants)?;
    let validation = timed_evaluation(best_variant, best.genome, EvaluationSet::Validation);
    let eval = final_eval(best_variant, best.genome)?;
    let best_genome = genome_report(best.genome);

    let artifact_path = args.out_path.clone();
    let mut history_path = None;
    if let Some(out_path) = args.out_path.as_deref() {
        save_evolved_artifact(
            out_path,
            &EvolvedPancArtifact {
                version: 1,
                model: "evolved_panc_like".to_string(),
                class_names: class_names.clone(),
                best_genome: best_genome.clone(),
                best_fitness: best.fitness,
                validation_accuracy: validation.evaluation.accuracy,
                eval_accuracy: eval.accuracy,
                seed: config.seed,
                population_size: config.population_size,
                generations: config.generations,
                elite_count: config.elite_count,
                mutation_rate: config.mutation_rate,
                validation_ratio: args.evolution_validation_ratio,
                threads: config.threads,
                hardware_note: hardware_note(),
            },
        )?;
        let history = sibling_with_suffix(out_path, "history.csv");
        if let Some(parent) = history.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let file = fs::File::create(&history)?;
        write_evolution_history_csv(file, &rows)?;
        history_path = Some(history.to_string_lossy().to_string());
    }

    Ok(CommandOutput::Evolution(EvolutionReport {
        model: "evolved_panc_like".to_string(),
        dataset: "image-folder".to_string(),
        artifact_path,
        history_path,
        seed: config.seed,
        population_size: config.population_size,
        generations: config.generations,
        elite_count: config.elite_count,
        mutation_rate: config.mutation_rate,
        validation_ratio: args.evolution_validation_ratio,
        threads: config.threads,
        hardware_note: hardware_note(),
        best_genome,
        best_fitness: best.fitness,
        validation_accuracy: validation.evaluation.accuracy,
        eval_accuracy: eval.accuracy,
        eval_per_class_accuracy: eval.per_class_accuracy,
        eval_confusion_matrix: eval.confusion_matrix,
        rows,
    }))
}

fn score_population(
    population: &[PancGenome],
    threads: usize,
    variants: &[VariantData],
    args: &Args,
) -> Vec<ScoredGenome<PancGenome>> {
    evaluate_population_parallel(population, threads, &|genome| {
        let Ok(variant) = variant_for(genome, variants) else {
            return f64::NEG_INFINITY;
        };
        let timed = timed_evaluation(variant, *genome, EvaluationSet::Validation);
        fitness(
            &timed.evaluation,
            timed.elapsed_ms,
            args.evolution_memory_penalty_per_mb,
            args.evolution_inference_penalty_per_ms,
        )
    })
}

fn next_generation(
    scored: &[ScoredGenome<PancGenome>],
    space: &PancGenomeSpace,
    config: EvolutionConfig,
    rng: &mut impl Rng,
) -> Vec<PancGenome> {
    let mut next = scored
        .iter()
        .take(config.elite_count)
        .map(|value| value.genome)
        .collect::<Vec<_>>();

    while next.len() < config.population_size {
        let left = tournament(scored, rng);
        let right = tournament(scored, rng);
        let mut child = left.crossover(right, rng);
        child.mutate(space, config.mutation_rate, rng);
        next.push(child);
    }
    next
}

fn tournament(scored: &[ScoredGenome<PancGenome>], rng: &mut impl Rng) -> PancGenome {
    let mut best = &scored[rng.gen_range(0..scored.len())];
    for _ in 0..2 {
        let candidate = &scored[rng.gen_range(0..scored.len())];
        if candidate.fitness > best.fitness {
            best = candidate;
        }
    }
    best.genome
}

fn fitness(
    evaluation: &PancBinaryEvaluation,
    elapsed_ms: u128,
    memory_penalty_per_mb: f64,
    inference_penalty_per_ms: f64,
) -> f64 {
    let memory_mb = evaluation.memory_bytes as f64 / (1024.0 * 1024.0);
    evaluation.accuracy
        - memory_mb * memory_penalty_per_mb.max(0.0)
        - elapsed_ms as f64 * inference_penalty_per_ms.max(0.0)
}

fn evaluate_generation_row(
    generation: usize,
    scored: &ScoredGenome<PancGenome>,
    variants: &[VariantData],
) -> Result<EvolutionGenerationRow, Box<dyn Error>> {
    let variant = variant_for(&scored.genome, variants)?;
    let timed = timed_evaluation(variant, scored.genome, EvaluationSet::Validation);
    let genome = genome_report(scored.genome);
    Ok(EvolutionGenerationRow {
        generation,
        best_fitness: scored.fitness,
        validation_accuracy: timed.evaluation.accuracy,
        validation_ms: timed.elapsed_ms,
        memory_bytes: timed.evaluation.memory_bytes,
        image_size: genome.image_size,
        image_features: genome.image_features,
        image_resize: genome.image_resize,
        threshold: genome.threshold,
        similarity: genome.similarity,
        jaccard_weight: genome.jaccard_weight,
        top_k: genome.top_k,
    })
}

fn final_eval(variant: &VariantData, genome: PancGenome) -> Result<FinalEval, Box<dyn Error>> {
    let Some(eval_samples) = variant.eval_samples.as_ref() else {
        return Ok(FinalEval::default());
    };
    let Some(eval_labels) = variant.eval_labels.as_ref() else {
        return Ok(FinalEval::default());
    };

    let evaluation = evaluate_panc_binary(
        &variant.library_samples,
        &variant.library_labels,
        eval_samples,
        eval_labels,
        variant.class_names.len(),
        genome,
    );
    let diagnostics =
        classification_metrics(eval_labels, &evaluation.predictions, &variant.class_names);
    Ok(FinalEval {
        accuracy: Some(diagnostics.accuracy),
        per_class_accuracy: diagnostics.per_class_accuracy,
        confusion_matrix: diagnostics.confusion_matrix,
    })
}

fn timed_evaluation(
    variant: &VariantData,
    genome: PancGenome,
    set: EvaluationSet,
) -> TimedEvaluation {
    let start = Instant::now();
    let (samples, labels) = match set {
        EvaluationSet::Validation => (&variant.validation_samples, &variant.validation_labels),
    };
    let evaluation = evaluate_panc_binary(
        &variant.library_samples,
        &variant.library_labels,
        samples,
        labels,
        variant.class_names.len(),
        genome,
    );
    TimedEvaluation {
        evaluation,
        elapsed_ms: start.elapsed().as_millis(),
    }
}

fn sort_scored(scored: &mut [ScoredGenome<PancGenome>]) {
    scored.sort_by(|left, right| {
        right
            .fitness
            .total_cmp(&left.fitness)
            .then_with(|| left.genome.image_size.cmp(&right.genome.image_size))
            .then_with(|| left.genome.top_k.cmp(&right.genome.top_k))
    });
}

fn genome_report(genome: PancGenome) -> EvolvedPancGenomeReport {
    EvolvedPancGenomeReport {
        image_size: genome.image_size,
        image_features: genome.feature_mode.as_str().to_string(),
        image_resize: genome.resize_mode.as_str().to_string(),
        threshold: genome.threshold,
        similarity: genome.similarity_name().to_string(),
        jaccard_weight: genome.jaccard_weight,
        top_k: genome.top_k,
    }
}

fn load_variants(
    data_path: &str,
    eval_path: Option<&str>,
    space: &PancGenomeSpace,
    args: &Args,
) -> Result<Vec<VariantData>, Box<dyn Error>> {
    let mut variants = Vec::new();
    for image_size in &space.image_sizes {
        for feature_mode in &space.feature_modes {
            for resize_mode in &space.resize_modes {
                let image_config = ImageVectorConfig::new(*image_size, *image_size)
                    .with_feature_mode(*feature_mode)
                    .with_resize_mode(*resize_mode);
                let dataset = load_image_folder(data_path, image_config)?;
                let eval_dataset = eval_path
                    .map(|path| load_image_folder(path, image_config))
                    .transpose()?;
                variants.push(prepare_variant(
                    *image_size,
                    *feature_mode,
                    *resize_mode,
                    dataset,
                    eval_dataset,
                    args,
                )?);
            }
        }
    }
    Ok(variants)
}

fn prepare_variant(
    image_size: u32,
    feature_mode: ImageFeatureMode,
    resize_mode: ImageResizeMode,
    dataset: Dataset,
    eval_dataset: Option<Dataset>,
    args: &Args,
) -> Result<VariantData, Box<dyn Error>> {
    let validation_ratio = args.evolution_validation_ratio.clamp(0.05, 0.5);
    let (validation_indexes, library_indexes) =
        train_test_split_indices(dataset.samples.len(), validation_ratio, args.seed);
    if validation_indexes.is_empty() || library_indexes.is_empty() {
        return Err("evolution requires both library and validation samples".into());
    }

    let library_samples = collect_samples(&dataset.samples, &library_indexes);
    let library_labels = collect_labels(&dataset.labels, &library_indexes);
    let validation_samples = collect_samples(&dataset.samples, &validation_indexes);
    let validation_labels = collect_labels(&dataset.labels, &validation_indexes);
    let ranges = min_max_ranges(&library_samples);
    let library_samples = min_max_scale(&library_samples, &ranges);
    let validation_samples = min_max_scale(&validation_samples, &ranges);

    let (eval_samples, eval_labels) = if let Some(eval_dataset) = eval_dataset {
        let labels = remap_labels_by_class_name(&eval_dataset, &dataset.class_names)?;
        (
            Some(min_max_scale(&eval_dataset.samples, &ranges)),
            Some(labels),
        )
    } else {
        (None, None)
    };

    Ok(VariantData {
        image_size,
        feature_mode,
        resize_mode,
        class_names: dataset.class_names,
        library_samples,
        library_labels,
        validation_samples,
        validation_labels,
        eval_samples,
        eval_labels,
    })
}

fn variant_for<'a>(
    genome: &PancGenome,
    variants: &'a [VariantData],
) -> Result<&'a VariantData, Box<dyn Error>> {
    variants
        .iter()
        .find(|variant| {
            variant.image_size == genome.image_size
                && variant.feature_mode == genome.feature_mode
                && variant.resize_mode == genome.resize_mode
        })
        .ok_or_else(|| "genome references a variant that was not precomputed".into())
}

fn collect_samples(samples: &[Vec<f64>], indexes: &[usize]) -> Vec<Vec<f64>> {
    indexes
        .iter()
        .map(|index| samples[*index].clone())
        .collect()
}

fn collect_labels(labels: &[usize], indexes: &[usize]) -> Vec<usize> {
    indexes.iter().map(|index| labels[*index]).collect()
}

fn remap_labels_by_class_name(
    source: &Dataset,
    target_class_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error>> {
    source
        .labels
        .iter()
        .map(|label| {
            let class_name = source
                .class_names
                .get(*label)
                .ok_or_else(|| format!("missing source class name for label {label}"))?;
            target_class_names
                .iter()
                .position(|target| target == class_name)
                .ok_or_else(|| {
                    format!("eval class {class_name:?} does not exist in training classes")
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn genome_space(args: &Args) -> PancGenomeSpace {
    let image_sizes = if args.evolution_image_sizes.is_empty() {
        if args.matrix_image_sizes.is_empty() {
            default_image_sizes(args)
        } else {
            args.matrix_image_sizes.clone()
        }
    } else {
        args.evolution_image_sizes.clone()
    };
    let feature_modes = if args.evolution_feature_modes.is_empty() {
        if args.matrix_features.is_empty() {
            default_feature_modes(args)
        } else {
            args.matrix_features.clone()
        }
    } else {
        args.evolution_feature_modes.clone()
    };
    let resize_modes = if args.evolution_resize_modes.is_empty() {
        if args.matrix_resize_modes.is_empty() {
            default_resize_modes(args)
        } else {
            args.matrix_resize_modes.clone()
        }
    } else {
        args.evolution_resize_modes.clone()
    };
    let top_k_values = if args.evolution_top_k_values.is_empty() {
        dedup_preserving_order(vec![args.top_k.max(1), 1, 3, 5, 7])
    } else {
        dedup_preserving_order(
            args.evolution_top_k_values
                .iter()
                .copied()
                .map(|value| value.max(1))
                .collect(),
        )
    };

    PancGenomeSpace {
        image_sizes: dedup_preserving_order(image_sizes),
        feature_modes: dedup_preserving_order(feature_modes),
        resize_modes: dedup_preserving_order(resize_modes),
        top_k_values,
        threshold_min: 0.05,
        threshold_max: 0.95,
        threshold_step: 0.15,
    }
    .normalized()
}

fn default_image_sizes(args: &Args) -> Vec<u32> {
    if args.image_width == 16 && args.image_height == 16 {
        vec![64]
    } else {
        vec![args.image_width.max(args.image_height)]
    }
}

fn default_feature_modes(args: &Args) -> Vec<ImageFeatureMode> {
    if args.image_features == ImageFeatureMode::Pixels {
        vec![ImageFeatureMode::RichTexture]
    } else {
        vec![args.image_features]
    }
}

fn default_resize_modes(args: &Args) -> Vec<ImageResizeMode> {
    if args.image_resize == ImageResizeMode::Stretch {
        vec![ImageResizeMode::CenterCrop]
    } else {
        vec![args.image_resize]
    }
}

fn dedup_preserving_order<T: Copy + PartialEq>(values: Vec<T>) -> Vec<T> {
    let mut deduped = Vec::new();
    for value in values {
        if !deduped.contains(&value) {
            deduped.push(value);
        }
    }
    deduped
}

fn save_evolved_artifact(
    path: impl AsRef<Path>,
    artifact: &EvolvedPancArtifact,
) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(artifact)?)?;
    Ok(())
}

fn sibling_with_suffix(path: impl AsRef<Path>, suffix: &str) -> PathBuf {
    let path = path.as_ref();
    let mut name = path
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "evolved-panc".to_string());
    name.push('.');
    name.push_str(suffix);
    path.with_file_name(name)
}

fn hardware_note() -> String {
    "CPU-parallel genetic search; use --threads 32 on the noted Intel i9-14900/32-thread workstation. The NVIDIA RTX 5090 is not used by this first CPU implementation.".to_string()
}

#[derive(Debug, Clone)]
struct VariantData {
    image_size: u32,
    feature_mode: ImageFeatureMode,
    resize_mode: ImageResizeMode,
    class_names: Vec<String>,
    library_samples: Vec<Vec<f64>>,
    library_labels: Vec<usize>,
    validation_samples: Vec<Vec<f64>>,
    validation_labels: Vec<usize>,
    eval_samples: Option<Vec<Vec<f64>>>,
    eval_labels: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
enum EvaluationSet {
    Validation,
}

#[derive(Debug)]
struct TimedEvaluation {
    evaluation: PancBinaryEvaluation,
    elapsed_ms: u128,
}

#[derive(Debug, Default)]
struct FinalEval {
    accuracy: Option<f64>,
    per_class_accuracy: Vec<super::PerClassAccuracy>,
    confusion_matrix: Vec<super::ConfusionRow>,
}

#[derive(Debug, Serialize)]
struct EvolvedPancArtifact {
    version: u32,
    model: String,
    class_names: Vec<String>,
    best_genome: EvolvedPancGenomeReport,
    best_fitness: f64,
    validation_accuracy: f64,
    eval_accuracy: Option<f64>,
    seed: u64,
    population_size: usize,
    generations: usize,
    elite_count: usize,
    mutation_rate: f64,
    validation_ratio: f64,
    threads: usize,
    hardware_note: String,
}
