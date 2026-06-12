mod research_bench;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = research_bench::parse_args()?;
    research_bench::print_config_report(&args);
    let output = research_bench::run(&args)?;
    research_bench::write_output(&output, args.format)?;
    Ok(())
}
