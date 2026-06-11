mod research_bench;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = research_bench::parse_args()?;
    let metrics = research_bench::run(&args)?;
    research_bench::write_metrics(&metrics, args.format)?;
    Ok(())
}
