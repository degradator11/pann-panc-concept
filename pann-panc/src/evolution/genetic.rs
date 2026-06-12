use std::thread;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub elite_count: usize,
    pub mutation_rate: f64,
    pub seed: u64,
    pub threads: usize,
}

impl EvolutionConfig {
    pub fn normalized(self) -> Self {
        let auto_threads = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let threads = if self.threads == 0 {
            auto_threads
        } else {
            self.threads
        };
        let population_size = self.population_size.max(2);
        Self {
            population_size,
            generations: self.generations.max(1),
            elite_count: self.elite_count.clamp(1, population_size - 1),
            mutation_rate: self.mutation_rate.clamp(0.0, 1.0),
            seed: self.seed,
            threads: threads.clamp(1, population_size),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredGenome<G> {
    pub genome: G,
    pub fitness: f64,
}

pub fn evaluate_population_parallel<G, F>(
    population: &[G],
    threads: usize,
    evaluate: &F,
) -> Vec<ScoredGenome<G>>
where
    G: Clone + Send + Sync,
    F: Fn(&G) -> f64 + Sync,
{
    if population.is_empty() {
        return Vec::new();
    }

    let thread_count = threads.clamp(1, population.len());
    let chunk_size = population.len().div_ceil(thread_count);
    let mut scored = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in population.chunks(chunk_size).enumerate() {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .enumerate()
                    .map(|(offset, genome)| {
                        let index = chunk_index * chunk_size + offset;
                        (
                            index,
                            ScoredGenome {
                                genome: genome.clone(),
                                fitness: evaluate(genome),
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }

        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("population worker panicked"))
            .collect::<Vec<_>>()
    });

    scored.sort_by_key(|(index, _)| *index);
    scored.into_iter().map(|(_, value)| value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_evaluation_preserves_population_order() {
        let population = vec![3, 1, 4, 2];
        let scored = evaluate_population_parallel(&population, 2, &|value| f64::from(*value * 2));

        assert_eq!(
            scored.iter().map(|value| value.genome).collect::<Vec<_>>(),
            population
        );
        assert_eq!(
            scored
                .iter()
                .map(|value| value.fitness as i32)
                .collect::<Vec<_>>(),
            vec![6, 2, 8, 4]
        );
    }
}
