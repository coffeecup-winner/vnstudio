use crate::automata::game_of_life::GameOfLife;

mod automata;
mod core;

fn main() {
    let mut automaton = GameOfLife::new();
    automaton.switch_to_lut();
}
