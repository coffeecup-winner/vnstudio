use crate::automata::game_of_life::{GameOfLife, GameOfLifeState};

mod automata;
mod core;

fn main() {
    let mut automaton = GameOfLife::new();
    automaton.switch_to_lut();
    automaton.set_state(1, 0, GameOfLifeState::Live);
    automaton.set_state(2, 1, GameOfLifeState::Live);
    automaton.set_state(0, 2, GameOfLifeState::Live);
    automaton.set_state(1, 2, GameOfLifeState::Live);
    automaton.set_state(2, 2, GameOfLifeState::Live);

    for iteration in 0..10 {
        println!("Iteration {}:", iteration);
        for y in -5..15 {
            for x in -5..15 {
                print!("{}", automaton.get_state(x, y));
            }
            println!();
        }
        println!();
        automaton.evaluate_next();
    }
}
