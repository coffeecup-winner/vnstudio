use std::marker::PhantomData;

use super::types::*;

pub struct RuleLUT<State: CellState, Neighborhood: CellNeighborhood<State>> {
    lut: Vec<State>,
    _phantom_data: PhantomData<Neighborhood>,
}

impl<State: CellState, Neighborhood: CellNeighborhood<State>> CellRuleEvaluator<State, Neighborhood>
    for RuleLUT<State, Neighborhood>
{
    fn evaluate(&self, state: State, neighbors: &Neighborhood) -> State {
        self.lut[Self::to_index(state, neighbors)]
    }
}

impl<State: CellState, Neighborhood: CellNeighborhood<State>> RuleLUT<State, Neighborhood> {
    pub fn compute(evaluator: &dyn CellRuleEvaluator<State, Neighborhood>) -> Self {
        let num_states = State::COUNT;
        let num_bits_per_state = usize::BITS - (num_states - 1).leading_zeros();

        let total_num_bits = num_bits_per_state as usize * (Neighborhood::NUM_CELLS as usize + 1);
        assert!(
            total_num_bits <= usize::BITS as usize,
            "The automaton rules are too large to be encoded in a LUT"
        );

        let size = 1 << total_num_bits;

        assert!(
            size < 64 * 1024 * 1024,
            "LUT size is too large for this automaton, investigate"
        );
        let mut lut = vec![State::default(); size];

        for (i, result) in lut.iter_mut().enumerate() {
            if let Some((state, neighbors)) = Self::from_index(i) {
                *result = evaluator.evaluate(state, &neighbors);
            }
        }

        Self {
            lut,
            _phantom_data: PhantomData,
        }
    }

    fn to_index(state: State, neighbors: &Neighborhood) -> usize {
        // If these are not made const by the compiler, it will be slow
        let num_states = State::COUNT;
        let num_bits_per_state = usize::BITS - (num_states - 1).leading_zeros();

        let mut index = Into::<u8>::into(state) as usize;
        for s in neighbors.neighbors() {
            index <<= num_bits_per_state;
            index |= Into::<u8>::into(*s) as usize;
        }

        index
    }

    fn from_index(mut index: usize) -> Option<(State, Neighborhood)> {
        // If these are not made const by the compiler, it will be slow
        let num_states = State::COUNT;
        let num_bits_per_state = usize::BITS - (num_states - 1).leading_zeros();

        let mut neighbors = Neighborhood::default();
        for i in (0..Neighborhood::NUM_CELLS as usize).rev() {
            let component = (index & ((1 << num_bits_per_state) - 1)) as u8;
            neighbors.neighbors_mut()[i] = component.try_into().ok()?;
            index >>= num_bits_per_state;
        }

        let state = (index as u8).try_into().ok()?;

        Some((state, neighbors))
    }
}
