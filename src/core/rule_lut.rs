use super::types::*;

pub struct RuleLUT<const NEIGHBORHOOD_SIZE: usize, State: CellState> {
    lut: Vec<State>,
}

impl<const NEIGHBORHOOD_SIZE: usize, State: CellState> CellRuleEvaluator<NEIGHBORHOOD_SIZE, State>
    for RuleLUT<NEIGHBORHOOD_SIZE, State>
{
    fn evaluate(&self, state: State, neighbors: &[State; NEIGHBORHOOD_SIZE]) -> State {
        self.lut[Self::to_index(state, neighbors)]
    }
}

impl<const NEIGHBORHOOD_SIZE: usize, State: CellState> RuleLUT<NEIGHBORHOOD_SIZE, State> {
    pub fn compute(evaluator: &dyn CellRuleEvaluator<NEIGHBORHOOD_SIZE, State>) -> Self {
        let num_states = State::NUM_STATES;
        let num_bits_per_state = u8::BITS - (num_states - 1).leading_zeros();

        let size = 1 << (num_bits_per_state as usize * (NEIGHBORHOOD_SIZE + 1));
        let mut lut = vec![State::default(); size];

        for (i, result) in lut.iter_mut().enumerate() {
            if let Some((state, neighbors)) = Self::from_index(i) {
                *result = evaluator.evaluate(state, &neighbors);
            }
        }

        Self { lut }
    }

    fn to_index(state: State, neighbors: &[State; NEIGHBORHOOD_SIZE]) -> usize {
        // If these are not made const by the compiler, it will be slow
        let num_states = State::NUM_STATES;
        let num_bits_per_state = u8::BITS - (num_states - 1).leading_zeros();

        let mut index = Into::<u8>::into(state) as usize;
        for s in neighbors {
            index <<= num_bits_per_state;
            index |= Into::<u8>::into(*s) as usize;
        }

        index
    }

    fn from_index(mut index: usize) -> Option<(State, [State; NEIGHBORHOOD_SIZE])> {
        // If these are not made const by the compiler, it will be slow
        let num_states = State::NUM_STATES;
        let num_bits_per_state = u8::BITS - (num_states - 1).leading_zeros();

        let mut neighbors: [State; NEIGHBORHOOD_SIZE] = [State::default(); NEIGHBORHOOD_SIZE];
        for i in (0..NEIGHBORHOOD_SIZE).rev() {
            let component = (index & ((1 << num_bits_per_state) - 1)) as u8;
            neighbors[i] = component.try_into().ok()?;
            index >>= num_bits_per_state;
        }

        let state = (index as u8).try_into().ok()?;

        Some((state, neighbors))
    }
}
