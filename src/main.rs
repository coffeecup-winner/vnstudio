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

fn main() -> Result<(), Box<dyn Error>> {
    if let Some(path) = std::env::args().nth(1) {
        let pattern = golly_loader::load_jvn29_rle(PathBuf::from(path))?;
        let mut automaton = VonNeumann::new();
        pattern.apply_to(&mut automaton);
        run_app(automaton)?;
    } else {
        let mut automaton = GameOfLife::new();
        seed_game_of_life(&mut automaton);
        run_app(automaton)?;
    }

    Ok(())
}
