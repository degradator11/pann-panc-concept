use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use progress_ai::evolution::{EvolutionConfig, ScoredGenome, evaluate_population_parallel};
use progress_ai::preprocess::train_test_split_indices;
use progress_ai::vision::{
    ImageDatasetEntry, ImageFeatureMode, ImageManifestDataset, ImageResizeMode,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::patch_scan::{
    load_patch_dataset, normal_class_name, run_patch_scan_report, validate_patch_args,
};
use super::{
    Args, CommandOutput, EvolvedPatchScanSearchArtifact, PatchScanEvolutionReport,
    PatchScanEvolutionRow, PatchScanImageResult, PatchScanRecipeReport, PatchScanReport,
    save_evolved_patch_scan_search_artifact, write_patch_evolution_history_csv,
};

pub fn run_evolve_panc_patch_scan(args: &Args) -> Result<CommandOutput, Box<dyn Error>> {
    validate_patch_args(args)?;
    let input = load_patch_dataset(args)?;
    let normal_class = normal_class_name(&input.loaded)?;
    let eval_entries = input
        .loaded
        .eval
        .as_ref()
        .ok_or("patch scan evolution requires an eval split")?
        .entries
        .clone();
    let (validation_indexes, holdout_indexes) = split_eval_indexes(
        &eval_entries,
        &normal_class,
        args.evolution_validation_ratio,
        args.seed,
    );
    if validation_indexes.is_empty() {
        return Err("patch scan evolution needs at least one validation image".into());
    }

    let space = patch_genome_space(args).normalized();
    let config = EvolutionConfig {
        population_size: args.evolution_population,
        generations: args.evolution_generations,
        elite_count: args.evolution_elite_count,
        mutation_rate: args.evolution_mutation_rate,
        seed: args.seed,
        threads: args.evolution_threads,
    }
    .normalized();
    eprintln!(
        "patch evolution: scoring population={} generations={} threads={} validation_images={} holdout_images={}",
        config.population_size,
        config.generations,
        config.threads,
        validation_indexes.len(),
        holdout_indexes.len()
    );

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut population = (0..config.population_size)
        .map(|_| PatchScanGenome::random(&space, &mut rng))
        .collect::<Vec<_>>();
    let live_history_path = args
        .out_path
        .as_deref()
        .map(|out_path| sibling_with_suffix(out_path, "history.csv"));
    let mut rows = Vec::new();
    let mut final_scored = Vec::new();

    for generation in 0..config.generations {
        let mut scored = score_patch_population(
            &population,
            config.threads,
            &input.loaded,
            input.dataset_config_path.as_deref(),
            input.data_path.as_deref(),
            &validation_indexes,
            args,
        );
        sort_patch_scored(&mut scored);
        let best = scored
            .first()
            .ok_or("patch evolution produced an empty scored population")?;
        let row = patch_generation_row(
            generation,
            best,
            &input.loaded,
            input.dataset_config_path.as_deref(),
            input.data_path.as_deref(),
            &validation_indexes,
            args,
        )?;
        eprintln!(
            "patch evolution: generation {}/{} val_auroc={:.4} val_f1={:.4} fitness={:.4} {} size={} patch={} stride={} top_frac={:.3} q={:.3}",
            generation + 1,
            config.generations,
            row.validation_auroc,
            row.validation_f1,
            row.best_fitness,
            row.image_features,
            row.image_size,
            row.patch_size,
            row.patch_stride,
            row.patch_score_fraction,
            row.threshold_quantile,
        );
        rows.push(row);
        if let Some(history_path) = &live_history_path {
            save_patch_history(history_path, &rows)?;
        }

        if generation + 1 == config.generations {
            final_scored = scored;
            break;
        }

        population = next_patch_generation(&scored, &space, config, &mut rng);
    }

    sort_patch_scored(&mut final_scored);
    let best = final_scored
        .first()
        .ok_or("patch evolution did not produce a final best genome")?;
    let validation_report = evaluate_patch_genome(
        best.genome,
        &input.loaded,
        input.dataset_config_path.as_deref(),
        input.data_path.as_deref(),
        &validation_indexes,
        args,
    )?;
    let holdout_source = if holdout_indexes.is_empty() {
        &validation_indexes
    } else {
        &holdout_indexes
    };
    let holdout_report = evaluate_patch_genome(
        best.genome,
        &input.loaded,
        input.dataset_config_path.as_deref(),
        input.data_path.as_deref(),
        holdout_source,
        args,
    )?;
    let best_recipe = patch_recipe_report(best.genome);

    let artifact_path = args.out_path.clone();
    let mut history_path = None;
    if let Some(out_path) = args.out_path.as_deref() {
        save_evolved_patch_scan_search_artifact(
            out_path,
            &EvolvedPatchScanSearchArtifact {
                version: 1,
                model: "evolved_panc_patch_scan".to_string(),
                dataset: input
                    .loaded
                    .name
                    .clone()
                    .unwrap_or_else(|| "image-manifest".to_string()),
                best_recipe: best_recipe.clone(),
                best_fitness: best.fitness,
                validation_accuracy: validation_report.image_accuracy,
                validation_auroc: validation_report.image_auroc,
                validation_f1: patch_f1(&validation_report.results),
                holdout_accuracy: holdout_report.image_accuracy,
                holdout_auroc: holdout_report.image_auroc,
                holdout_f1: patch_f1(&holdout_report.results),
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
        save_patch_history(&history, &rows)?;
        history_path = Some(history.to_string_lossy().to_string());
    }

    Ok(CommandOutput::PatchEvolution(PatchScanEvolutionReport {
        model: "evolved_panc_patch_scan".to_string(),
        dataset: input
            .loaded
            .name
            .clone()
            .unwrap_or_else(|| "image-manifest".to_string()),
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
        best_recipe,
        best_fitness: best.fitness,
        validation_accuracy: validation_report.image_accuracy,
        validation_auroc: validation_report.image_auroc,
        validation_f1: patch_f1(&validation_report.results),
        holdout_accuracy: holdout_report.image_accuracy,
        holdout_auroc: holdout_report.image_auroc,
        holdout_f1: patch_f1(&holdout_report.results),
        rows,
    }))
}

fn score_patch_population(
    population: &[PatchScanGenome],
    threads: usize,
    loaded: &ImageManifestDataset,
    dataset_config_path: Option<&str>,
    data_path: Option<&str>,
    validation_indexes: &[usize],
    args: &Args,
) -> Vec<ScoredGenome<PatchScanGenome>> {
    evaluate_population_parallel(population, threads, &|genome| {
        let Ok(report) = evaluate_patch_genome(
            *genome,
            loaded,
            dataset_config_path,
            data_path,
            validation_indexes,
            args,
        ) else {
            return f64::NEG_INFINITY;
        };
        patch_fitness(&report, args)
    })
}

fn patch_generation_row(
    generation: usize,
    scored: &ScoredGenome<PatchScanGenome>,
    loaded: &ImageManifestDataset,
    dataset_config_path: Option<&str>,
    data_path: Option<&str>,
    validation_indexes: &[usize],
    args: &Args,
) -> Result<PatchScanEvolutionRow, Box<dyn Error>> {
    let report = evaluate_patch_genome(
        scored.genome,
        loaded,
        dataset_config_path,
        data_path,
        validation_indexes,
        args,
    )?;
    let recipe = patch_recipe_report(scored.genome);
    Ok(PatchScanEvolutionRow {
        generation,
        best_fitness: scored.fitness,
        validation_accuracy: report.image_accuracy,
        validation_auroc: report.image_auroc,
        validation_f1: patch_f1(&report.results),
        train_ms: report.train_ms,
        inference_ms: report.inference_ms,
        memory_bytes: report.memory_bytes,
        image_size: recipe.image_size,
        image_features: recipe.image_features,
        image_resize: recipe.image_resize,
        patch_size: recipe.patch_size,
        patch_stride: recipe.patch_stride,
        max_train_patches: recipe.max_train_patches,
        top_k: recipe.top_k,
        threshold_quantile: recipe.threshold_quantile,
        patch_score_fraction: recipe.patch_score_fraction,
    })
}

fn evaluate_patch_genome(
    genome: PatchScanGenome,
    loaded: &ImageManifestDataset,
    dataset_config_path: Option<&str>,
    data_path: Option<&str>,
    eval_indexes: &[usize],
    args: &Args,
) -> Result<PatchScanReport, Box<dyn Error>> {
    let genome_args = patch_args_from_genome(args, genome);
    run_patch_scan_report(
        loaded,
        dataset_config_path.map(str::to_string),
        data_path.map(str::to_string),
        Some(eval_indexes),
        &genome_args,
    )
}

fn patch_args_from_genome(args: &Args, genome: PatchScanGenome) -> Args {
    let mut next = args.clone();
    next.command = "panc-patch-scan".to_string();
    next.image_width = genome.image_size;
    next.image_height = genome.image_size;
    next.image_features = genome.feature_mode;
    next.image_resize = genome.resize_mode;
    next.patch_size = genome.patch_size;
    next.patch_stride = genome.patch_stride;
    next.max_train_patches = genome.max_train_patches;
    next.top_k = genome.top_k;
    next.anomaly_threshold_quantile = genome.threshold_quantile;
    next.patch_score_fraction = genome.patch_score_fraction;
    next.out_path = None;
    next.report_out_path = None;
    next.debug_out_path = None;
    next
}

fn patch_fitness(report: &PatchScanReport, args: &Args) -> f64 {
    let memory_mb = report.memory_bytes as f64 / (1024.0 * 1024.0);
    let elapsed_ms = report.train_ms.saturating_add(report.inference_ms);
    report.image_auroc * 0.65 + patch_f1(&report.results) * 0.25 + report.image_accuracy * 0.10
        - memory_mb * args.evolution_memory_penalty_per_mb.max(0.0)
        - elapsed_ms as f64 * args.evolution_inference_penalty_per_ms.max(0.0)
}

fn patch_f1(results: &[PatchScanImageResult]) -> f64 {
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    for result in results {
        match (result.expected_anomaly, result.predicted_anomaly) {
            (true, true) => true_positive += 1,
            (false, true) => false_positive += 1,
            (true, false) => false_negative += 1,
            (false, false) => {}
        }
    }
    let precision = if true_positive + false_positive == 0 {
        0.0
    } else {
        true_positive as f64 / (true_positive + false_positive) as f64
    };
    let recall = if true_positive + false_negative == 0 {
        0.0
    } else {
        true_positive as f64 / (true_positive + false_negative) as f64
    };
    if precision + recall <= f64::EPSILON {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn next_patch_generation(
    scored: &[ScoredGenome<PatchScanGenome>],
    space: &PatchScanGenomeSpace,
    config: EvolutionConfig,
    rng: &mut impl Rng,
) -> Vec<PatchScanGenome> {
    let mut next = scored
        .iter()
        .take(config.elite_count)
        .map(|value| value.genome)
        .collect::<Vec<_>>();

    while next.len() < config.population_size {
        let left = patch_tournament(scored, rng);
        let right = patch_tournament(scored, rng);
        let mut child = left.crossover(right, rng);
        child.mutate(space, config.mutation_rate, rng);
        next.push(child);
    }
    next
}

fn patch_tournament(
    scored: &[ScoredGenome<PatchScanGenome>],
    rng: &mut impl Rng,
) -> PatchScanGenome {
    let mut best = &scored[rng.gen_range(0..scored.len())];
    for _ in 0..2 {
        let candidate = &scored[rng.gen_range(0..scored.len())];
        if candidate.fitness > best.fitness {
            best = candidate;
        }
    }
    best.genome
}

fn sort_patch_scored(scored: &mut [ScoredGenome<PatchScanGenome>]) {
    scored.sort_by(|left, right| {
        right
            .fitness
            .total_cmp(&left.fitness)
            .then_with(|| left.genome.image_size.cmp(&right.genome.image_size))
            .then_with(|| left.genome.patch_size.cmp(&right.genome.patch_size))
            .then_with(|| left.genome.patch_stride.cmp(&right.genome.patch_stride))
    });
}

fn patch_recipe_report(genome: PatchScanGenome) -> PatchScanRecipeReport {
    PatchScanRecipeReport {
        image_size: genome.image_size,
        image_features: genome.feature_mode.as_str().to_string(),
        image_resize: genome.resize_mode.as_str().to_string(),
        patch_size: genome.patch_size,
        patch_stride: genome.patch_stride,
        max_train_patches: genome.max_train_patches,
        top_k: genome.top_k,
        threshold_quantile: genome.threshold_quantile,
        patch_score_fraction: genome.patch_score_fraction,
    }
}

fn split_eval_indexes(
    entries: &[ImageDatasetEntry],
    normal_class: &str,
    validation_ratio: f64,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut normal = Vec::new();
    let mut anomaly = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry.class_name == normal_class {
            normal.push(index);
        } else {
            anomaly.push(index);
        }
    }

    let (normal_validation, normal_holdout) =
        split_index_subset(&normal, validation_ratio, seed.wrapping_add(1));
    let (anomaly_validation, anomaly_holdout) =
        split_index_subset(&anomaly, validation_ratio, seed.wrapping_add(2));
    let mut validation = normal_validation
        .into_iter()
        .chain(anomaly_validation)
        .collect::<Vec<_>>();
    let mut holdout = normal_holdout
        .into_iter()
        .chain(anomaly_holdout)
        .collect::<Vec<_>>();
    validation.sort_unstable();
    holdout.sort_unstable();
    (validation, holdout)
}

fn split_index_subset(indexes: &[usize], ratio: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    if indexes.len() <= 1 {
        return (indexes.to_vec(), Vec::new());
    }
    let (validation_offsets, holdout_offsets) =
        train_test_split_indices(indexes.len(), ratio.clamp(0.05, 0.8), seed);
    let validation = validation_offsets
        .into_iter()
        .filter_map(|offset| indexes.get(offset).copied())
        .collect::<Vec<_>>();
    let holdout = holdout_offsets
        .into_iter()
        .filter_map(|offset| indexes.get(offset).copied())
        .collect::<Vec<_>>();
    (validation, holdout)
}

fn save_patch_history(
    path: impl AsRef<Path>,
    rows: &[PatchScanEvolutionRow],
) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    write_patch_evolution_history_csv(file, rows)?;
    Ok(())
}

fn sibling_with_suffix(path: impl AsRef<Path>, suffix: &str) -> PathBuf {
    let path = path.as_ref();
    let mut name = path
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| "evolved-panc-patch-scan".to_string());
    name.push('.');
    name.push_str(suffix);
    path.with_file_name(name)
}

fn hardware_note() -> String {
    "CPU-parallel genetic search for patch-scan recipes; use --threads 32 on the noted Intel i9-14900/32-thread workstation. The NVIDIA RTX 5090 is not used by this CPU implementation.".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PatchScanGenome {
    image_size: u32,
    feature_mode: ImageFeatureMode,
    resize_mode: ImageResizeMode,
    patch_size: u32,
    patch_stride: u32,
    max_train_patches: usize,
    top_k: usize,
    threshold_quantile: f64,
    patch_score_fraction: f64,
}

impl PatchScanGenome {
    fn random(space: &PatchScanGenomeSpace, rng: &mut impl Rng) -> Self {
        Self {
            image_size: pick(&space.image_sizes, rng),
            feature_mode: pick(&space.feature_modes, rng),
            resize_mode: pick(&space.resize_modes, rng),
            patch_size: pick(&space.patch_sizes, rng),
            patch_stride: pick(&space.patch_strides, rng),
            max_train_patches: pick(&space.max_train_patches, rng),
            top_k: pick(&space.top_k_values, rng),
            threshold_quantile: pick(&space.threshold_quantiles, rng),
            patch_score_fraction: pick(&space.patch_score_fractions, rng),
        }
    }

    fn crossover(self, other: Self, rng: &mut impl Rng) -> Self {
        Self {
            image_size: pick_parent(self.image_size, other.image_size, rng),
            feature_mode: pick_parent(self.feature_mode, other.feature_mode, rng),
            resize_mode: pick_parent(self.resize_mode, other.resize_mode, rng),
            patch_size: pick_parent(self.patch_size, other.patch_size, rng),
            patch_stride: pick_parent(self.patch_stride, other.patch_stride, rng),
            max_train_patches: pick_parent(self.max_train_patches, other.max_train_patches, rng),
            top_k: pick_parent(self.top_k, other.top_k, rng),
            threshold_quantile: pick_parent(self.threshold_quantile, other.threshold_quantile, rng),
            patch_score_fraction: pick_parent(
                self.patch_score_fraction,
                other.patch_score_fraction,
                rng,
            ),
        }
    }

    fn mutate(&mut self, space: &PatchScanGenomeSpace, mutation_rate: f64, rng: &mut impl Rng) {
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
            self.patch_size = pick(&space.patch_sizes, rng);
        }
        if rng.gen_bool(rate) {
            self.patch_stride = pick(&space.patch_strides, rng);
        }
        if rng.gen_bool(rate) {
            self.max_train_patches = pick(&space.max_train_patches, rng);
        }
        if rng.gen_bool(rate) {
            self.top_k = pick(&space.top_k_values, rng);
        }
        if rng.gen_bool(rate) {
            self.threshold_quantile = pick(&space.threshold_quantiles, rng);
        }
        if rng.gen_bool(rate) {
            self.patch_score_fraction = pick(&space.patch_score_fractions, rng);
        }
    }
}

#[derive(Debug, Clone)]
struct PatchScanGenomeSpace {
    image_sizes: Vec<u32>,
    feature_modes: Vec<ImageFeatureMode>,
    resize_modes: Vec<ImageResizeMode>,
    patch_sizes: Vec<u32>,
    patch_strides: Vec<u32>,
    max_train_patches: Vec<usize>,
    top_k_values: Vec<usize>,
    threshold_quantiles: Vec<f64>,
    patch_score_fractions: Vec<f64>,
}

impl PatchScanGenomeSpace {
    fn normalized(mut self) -> Self {
        if self.image_sizes.is_empty() {
            self.image_sizes.push(16);
        }
        if self.feature_modes.is_empty() {
            self.feature_modes.push(ImageFeatureMode::RichEdge);
        }
        if self.resize_modes.is_empty() {
            self.resize_modes.push(ImageResizeMode::Stretch);
        }
        if self.patch_sizes.is_empty() {
            self.patch_sizes.push(160);
        }
        if self.patch_strides.is_empty() {
            self.patch_strides.push(80);
        }
        if self.max_train_patches.is_empty() {
            self.max_train_patches.push(512);
        }
        if self.top_k_values.is_empty() {
            self.top_k_values.push(1);
        }
        if self.threshold_quantiles.is_empty() {
            self.threshold_quantiles.push(0.95);
        }
        if self.patch_score_fractions.is_empty() {
            self.patch_score_fractions.push(0.10);
        }
        self.image_sizes = dedup_preserving_order(
            self.image_sizes
                .into_iter()
                .filter(|value| *value > 0)
                .collect(),
        );
        self.patch_sizes = dedup_preserving_order(
            self.patch_sizes
                .into_iter()
                .filter(|value| *value > 0)
                .collect(),
        );
        self.patch_strides = dedup_preserving_order(
            self.patch_strides
                .into_iter()
                .filter(|value| *value > 0)
                .collect(),
        );
        self.max_train_patches = dedup_preserving_order(
            self.max_train_patches
                .into_iter()
                .filter(|value| *value > 0)
                .collect(),
        );
        self.top_k_values = dedup_preserving_order(
            self.top_k_values
                .into_iter()
                .map(|value| value.max(1))
                .collect(),
        );
        self.threshold_quantiles = dedup_f64_preserving_order(
            self.threshold_quantiles
                .into_iter()
                .map(|value| value.clamp(0.0, 1.0))
                .collect(),
        );
        self.patch_score_fractions = dedup_f64_preserving_order(
            self.patch_score_fractions
                .into_iter()
                .map(|value| value.clamp(0.01, 1.0))
                .collect(),
        );
        if self.image_sizes.is_empty() {
            self.image_sizes.push(16);
        }
        if self.patch_sizes.is_empty() {
            self.patch_sizes.push(160);
        }
        if self.patch_strides.is_empty() {
            self.patch_strides.push(80);
        }
        if self.max_train_patches.is_empty() {
            self.max_train_patches.push(512);
        }
        if self.threshold_quantiles.is_empty() {
            self.threshold_quantiles.push(0.95);
        }
        if self.patch_score_fractions.is_empty() {
            self.patch_score_fractions.push(0.10);
        }
        self
    }
}

fn patch_genome_space(args: &Args) -> PatchScanGenomeSpace {
    PatchScanGenomeSpace {
        image_sizes: if args.evolution_image_sizes.is_empty() {
            dedup_preserving_order(vec![args.image_width.max(args.image_height).max(1), 16, 24])
        } else {
            args.evolution_image_sizes.clone()
        },
        feature_modes: if args.evolution_feature_modes.is_empty() {
            dedup_preserving_order(vec![
                args.image_features,
                ImageFeatureMode::RichEdge,
                ImageFeatureMode::RichTexture,
                ImageFeatureMode::RichHog,
            ])
        } else {
            args.evolution_feature_modes.clone()
        },
        resize_modes: if args.evolution_resize_modes.is_empty() {
            dedup_preserving_order(vec![args.image_resize])
        } else {
            args.evolution_resize_modes.clone()
        },
        patch_sizes: if args.evolution_patch_sizes.is_empty() {
            dedup_preserving_order(vec![args.patch_size, 96, 128, 160, 224])
        } else {
            args.evolution_patch_sizes.clone()
        },
        patch_strides: if args.evolution_patch_strides.is_empty() {
            dedup_preserving_order(vec![args.patch_stride, 48, 64, 80, 112, 160])
        } else {
            args.evolution_patch_strides.clone()
        },
        max_train_patches: if args.evolution_max_train_patches.is_empty() {
            if args.max_train_patches == 4096 {
                vec![256, 512, 1024]
            } else {
                dedup_preserving_order(vec![args.max_train_patches, 256, 512, 1024])
            }
        } else {
            args.evolution_max_train_patches.clone()
        },
        top_k_values: if args.evolution_top_k_values.is_empty() {
            dedup_preserving_order(vec![args.top_k.max(1), 1, 3, 5])
        } else {
            args.evolution_top_k_values.clone()
        },
        threshold_quantiles: if args.evolution_threshold_quantiles.is_empty() {
            dedup_f64_preserving_order(vec![
                args.anomaly_threshold_quantile,
                0.80,
                0.85,
                0.90,
                0.95,
            ])
        } else {
            args.evolution_threshold_quantiles.clone()
        },
        patch_score_fractions: if args.evolution_patch_score_fractions.is_empty() {
            dedup_f64_preserving_order(vec![args.patch_score_fraction, 0.03, 0.05, 0.10, 0.20])
        } else {
            args.evolution_patch_score_fractions.clone()
        },
    }
}

fn pick<T: Copy>(values: &[T], rng: &mut impl Rng) -> T {
    values[rng.gen_range(0..values.len())]
}

fn pick_parent<T: Copy>(left: T, right: T, rng: &mut impl Rng) -> T {
    if rng.gen_bool(0.5) { left } else { right }
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

fn dedup_f64_preserving_order(values: Vec<f64>) -> Vec<f64> {
    let mut deduped = Vec::new();
    for value in values {
        if !deduped
            .iter()
            .any(|existing: &f64| (*existing - value).abs() <= f64::EPSILON)
        {
            deduped.push(value);
        }
    }
    deduped
}
