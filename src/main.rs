use crate::automata::game_of_life::GameOfLife;

mod automata;
mod core;

fn main() {
    let _ = GameOfLife::new();

    println!("Hello, world!");
}
