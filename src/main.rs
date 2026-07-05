mod automata;
mod core;
mod gui;

use std::{error::Error, path::PathBuf};

use gui::app::VnStudioApp;

use crate::{
    automata::{
        game_of_life::{GameOfLife, GameOfLifeState},
        von_neumann::VonNeumann,
    },
    core::{
        cuda_evaluator::CudaEvaluator,
        golly_loader,
        storage::{Chunk, FillNeighborhood},
        types::{CellularAutomataConfig, CellularAutomaton},
    },
};

fn seed_game_of_life(automaton: &mut GameOfLife) {
    automaton.set_state(1, 0, GameOfLifeState::Live);
    automaton.set_state(2, 1, GameOfLifeState::Live);
    automaton.set_state(0, 2, GameOfLifeState::Live);
    automaton.set_state(1, 2, GameOfLifeState::Live);
    automaton.set_state(2, 2, GameOfLifeState::Live);
}

fn run_app<Config>(automaton: CellularAutomaton<Config>) -> eframe::Result<()>
where
    Config: CellularAutomataConfig,
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "VNStudio",
        options,
        Box::new(move |creation_context| {
            Ok(Box::new(VnStudioApp::new(creation_context, automaton)))
        }),
    )
}

fn benchmark<Config>(mut automaton: CellularAutomaton<Config>)
where
    Config: CellularAutomataConfig,
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    println!("Warming up");
    for _ in 0..1000 {
        automaton.evaluate_next();
    }
    println!("Starting the benchmark");
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        automaton.evaluate_next();
    }
    let end = std::time::Instant::now();
    let total = end - start;
    println!(
        "Iterating 10000 times took {}ms, {} UPS",
        total.as_millis(),
        (10_000_000f64 / total.as_millis() as f64) as u64
    );

    let times = automaton.operation_times();
    let total_ops =
        times.total_grid_evaluate + times.total_storage_apply + times.total_storage_optimize;
    println!("Total operations: {}ms", total_ops.as_millis());
    println!(
        "Total grid evaluation: {}ms ({:.2}%)",
        times.total_grid_evaluate.as_millis(),
        times.total_grid_evaluate.as_millis() as f64 * 100.0 / total_ops.as_millis() as f64
    );
    println!(
        "Total storage update: {}ms ({:.2}%)",
        times.total_storage_apply.as_millis(),
        times.total_storage_apply.as_millis() as f64 * 100.0 / total_ops.as_millis() as f64
    );
    println!(
        "Total storage optimization: {}ms ({:.2}%)",
        times.total_storage_optimize.as_millis(),
        times.total_storage_optimize.as_millis() as f64 * 100.0 / total_ops.as_millis() as f64
    );
    automaton.print_evaluator_stats();
}

fn cuda_enabled() -> bool {
    std::env::var("VNSTUDIO_CUDA").as_deref() == Ok("1")
}

fn new_von_neumann() -> Result<VonNeumann, Box<dyn Error>> {
    if cuda_enabled() {
        println!("Using CUDA evaluator");
        VonNeumann::try_new_with_grid_evaluator(|lut| {
            Ok(Box::new(CudaEvaluator::new(lut.values().to_vec())?))
        })
    } else {
        Ok(VonNeumann::new())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if let Some(path) = std::env::args().nth(1) {
        let pattern = golly_loader::load_jvn29_rle(PathBuf::from(path))?;
        let mut automaton = new_von_neumann()?;
        pattern.apply_to(&mut automaton);

        if let Some(arg) = std::env::args().nth(2)
            && arg == "--bench"
        {
            benchmark(automaton);
            return Ok(());
        }

        run_app(automaton)?;
    } else {
        let mut automaton = GameOfLife::new();
        seed_game_of_life(&mut automaton);
        run_app(automaton)?;
    }

    Ok(())
}
