mod research_bench;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = research_bench::parse_args()?;
    research_bench::print_config_report(&args);
    let output = research_bench::run(&args)?;
    if let Some(path) = args.report_out_path.as_deref() {
        research_bench::save_output_json(path, &output)?;
    }
    research_bench::write_output(&output, args.format)?;
    Ok(())
}
