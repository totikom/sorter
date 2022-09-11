use dbg_pls::{color, pretty, DebugPls, Formatter};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Zip};
use std::f64::consts::TAU;
use std::fmt;

fn main() {
    println!("Hello, world!");
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct TrapParams<const WIDTH: usize, const HEIGHT: usize> {
    x_frequencies: [f64; WIDTH],     // in Hz
    y_frequencies: [f64; HEIGHT],    // in Hz
    turn_on_time: f64,               // is seconds, time to turn on the laser
    local_oscillator_frequency: f64, // in Hz
    signal_amplitude: f64,           // amplitude of one harmonic
    buff_size: usize,                // Size of the SDR buffer
    sample_rate: f64,                // Sample rate of the SDR
    atom_speed: f64,                 // in Hz/s
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Trap<const WIDTH: usize, const HEIGHT: usize>([[bool; WIDTH]; HEIGHT]);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ShiftedTrap<const WIDTH: usize, const HEIGHT: usize> {
    array: [[bool; WIDTH]; HEIGHT],
    filled_trap_count: usize,
    target_size: usize,
    start_index: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct HorizontalMove {
    line_index: usize,
    moves: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct VerticalMove {
    line_index: usize,
    moves: Vec<(usize, usize)>,
}

impl<const WIDTH: usize, const HEIGHT: usize> Trap<WIDTH, HEIGHT> {
    fn shift(self) -> (Vec<HorizontalMove>, ShiftedTrap<WIDTH, HEIGHT>) {
        let mut shifted_trap = [[false; WIDTH]; HEIGHT];

        let filled_trap_count = self.0.iter().fold(0, |acc, line| {
            acc + line.iter().fold(0, |acc, x| if *x { acc + 1 } else { acc })
        });

        let target_size: usize = f64::sqrt(filled_trap_count as f64).floor() as usize;

        let start_index: usize = (WIDTH - target_size) / 2;

        let mut pointer = start_index;

        let moves = self
            .0
            .iter()
            .zip(shifted_trap.iter_mut())
            .enumerate()
            .map(|(i, (line, shifted_line))| {
                let mut sum = line.iter().fold(0, |acc, x| if *x { acc + 1 } else { acc });

                let mut breaked = false;
                for j in pointer..start_index + target_size {
                    if sum > 0 {
                        shifted_line[j] = true;
                        sum -= 1;
                    } else {
                        pointer = j;
                        breaked = true;
                        break;
                    }
                }

                if sum > 0 {
                    for j in start_index..pointer {
                        if sum > 0 {
                            shifted_line[j] = true;
                            sum -= 1;
                        } else {
                            pointer = j;
                            breaked = true;
                            break;
                        }
                    }

                    if sum > 0 {
                        for j in start_index + target_size..WIDTH {
                            if sum > 0 {
                                shifted_line[j] = true;
                                sum -= 1;
                            } else {
                                break;
                            }
                        }
                    }

                    if sum > 0 {
                        for j in (0..start_index).rev() {
                            if sum > 0 {
                                shifted_line[j] = true;
                                sum -= 1;
                            } else {
                                break;
                            }
                        }
                    }
                } else if !breaked {
                    pointer = start_index;
                }

                let start_iterator = line
                    .iter()
                    .enumerate()
                    .map(|(i, is_full)| if *is_full { Some(i) } else { None })
                    .flatten();
                let end_iterator = shifted_line
                    .iter()
                    .enumerate()
                    .map(|(i, is_full)| if *is_full { Some(i) } else { None })
                    .flatten();

                let line_moves = start_iterator.zip(end_iterator).collect();

                HorizontalMove {
                    line_index: i,
                    moves: line_moves,
                }
            })
            .collect();

        let shifted_trap = ShiftedTrap {
            array: shifted_trap,
            filled_trap_count,
            start_index,
            target_size,
        };

        (moves, shifted_trap)
    }

    fn from_nums(array: &[[u8; WIDTH]; HEIGHT]) -> Self {
        let mut new_array = [[false; WIDTH]; HEIGHT];
        for (i, row) in new_array.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                if array[i][j] != 0 {
                    *val = true;
                }
            }
        }

        Self(new_array)
    }

    fn to_nums(&self) -> [[u8; WIDTH]; HEIGHT] {
        let mut new_array = [[0; WIDTH]; HEIGHT];
        for (i, row) in new_array.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                if self.0[i][j] {
                    *val = 1;
                }
            }
        }

        new_array
    }

    fn atom_count(&self) -> usize {
        let mut sum = 0;
        for row in self.0 {
            for val in row {
                if val {
                    sum += 1;
                }
            }
        }
        sum
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> ShiftedTrap<WIDTH, HEIGHT> {
    fn merge(self) -> (Vec<VerticalMove>, Trap<WIDTH, HEIGHT>) {
        let mut trap = [[false; WIDTH]; HEIGHT];

        let moves = (0..WIDTH)
            .into_iter()
            .map(|j| {
                let mut sum = 0;
                let mut start_iterator = Vec::new();
                let mut end_iterator = Vec::new();

                for i in 0..HEIGHT {
                    if self.array[i][j] {
                        sum += 1;
                        start_iterator.push(i);
                    }
                }

                let start = (HEIGHT - sum) / 2;
                let end = start + sum;

                for i in start..end {
                    end_iterator.push(i);
                    trap[i][j] = true;
                }
                let line_moves = start_iterator
                    .into_iter()
                    .zip(end_iterator.into_iter())
                    .collect();

                VerticalMove {
                    line_index: j,
                    moves: line_moves,
                }
            })
            .collect();

        (moves, Trap(trap))
    }

    #[cfg(test)]
    fn get_trap(self) -> Trap<WIDTH, HEIGHT> {
        Trap(self.array)
    }
}

struct Signal {
    x_signal_i: Array1<f64>,
    x_signal_q: Array1<f64>,
    y_signal_i: Array1<f64>,
    y_signal_q: Array1<f64>,
}

impl<const WIDTH: usize, const HEIGHT: usize> TrapParams<WIDTH, HEIGHT> {
    fn generate_horizontal_move(&self, mov: HorizontalMove) -> Signal {
        let line_freq = self.y_frequencies[mov.line_index] - self.local_oscillator_frequency;

        let mut y_signal_i = Array1::<f64>::zeros(self.buff_size);
        let mut y_signal_q = Array1::<f64>::zeros(self.buff_size);

        Zip::indexed(&mut y_signal_i).par_for_each(|i, y| {
            *y = self.signal_amplitude * (i as f64 / self.sample_rate * TAU * line_freq).cos();
        });

        Zip::indexed(&mut y_signal_q).par_for_each(|i, y| {
            *y = self.signal_amplitude * (i as f64 / self.sample_rate * TAU * line_freq).sin();
        });

        let mut x_signal_i = Array1::<f64>::zeros(self.buff_size);
        let mut x_signal_q = Array1::<f64>::zeros(self.buff_size);

        for (start_idx, end_idx) in mov.moves {
            let start_freq_prepared =
                (self.x_frequencies[start_idx] - self.local_oscillator_frequency) * TAU
                    / self.sample_rate;
            let end_freq_prepared = (self.x_frequencies[end_idx] - self.local_oscillator_frequency)
                * TAU
                / self.sample_rate;
            let amplitude_prepared = self.signal_amplitude / self.turn_on_time;

            let freq_diff = end_freq_prepared - start_freq_prepared;

            let atom_speed_prepared = if freq_diff > 0.0 {
                self.atom_speed * TAU / self.sample_rate
            } else {
                -self.atom_speed * TAU / self.sample_rate
            };
            let move_time = self.turn_on_time + freq_diff.abs() / atom_speed_prepared;

            Zip::indexed(&mut x_signal_i).par_for_each(|i, x| {
                let t = i as f64 / self.sample_rate;
                if t < self.turn_on_time {
                    let amplitude = amplitude_prepared * t;
                    *x += amplitude * (i as f64 * start_freq_prepared).cos();
                } else if t < move_time {
                    let freq = start_freq_prepared + atom_speed_prepared * (t - self.turn_on_time);
                    *x += self.signal_amplitude * (i as f64 * freq).cos();
                } else if t < move_time + self.turn_on_time {
                    let amplitude = amplitude_prepared * (move_time + self.turn_on_time - t);
                    *x += amplitude * (i as f64 * end_freq_prepared).cos();
                }
            });

            Zip::indexed(&mut x_signal_q).par_for_each(|i, x| {
                let t = i as f64 / self.sample_rate;
                if t < self.turn_on_time {
                    let amplitude = amplitude_prepared * t;
                    *x += amplitude * (i as f64 * start_freq_prepared).sin();
                } else if t < move_time {
                    let freq = start_freq_prepared + atom_speed_prepared * (t - self.turn_on_time);
                    *x += self.signal_amplitude * (i as f64 * freq).sin();
                } else if t < move_time + self.turn_on_time {
                    let amplitude = amplitude_prepared * (move_time + self.turn_on_time - t);
                    *x += amplitude * (i as f64 * end_freq_prepared).sin();
                }
            });
        }

        Signal {
            x_signal_i,
            x_signal_q,
            y_signal_i,
            y_signal_q,
        }
    }

    fn generate_vertical_move(&self, mov: VerticalMove) -> Signal {
        let line_freq = self.x_frequencies[mov.line_index] - self.local_oscillator_frequency;

        let mut x_signal_i = Array1::<f64>::zeros(self.buff_size);
        let mut x_signal_q = Array1::<f64>::zeros(self.buff_size);

        Zip::indexed(&mut x_signal_i).par_for_each(|i, x| {
            *x = self.signal_amplitude * (i as f64 / self.sample_rate * TAU * line_freq).cos();
        });

        Zip::indexed(&mut x_signal_q).par_for_each(|i, x| {
            *x = self.signal_amplitude * (i as f64 / self.sample_rate * TAU * line_freq).sin();
        });

        let mut y_signal_i = Array1::<f64>::zeros(self.buff_size);
        let mut y_signal_q = Array1::<f64>::zeros(self.buff_size);

        for (start_idx, end_idx) in mov.moves {
            let start_freq_prepared =
                (self.y_frequencies[start_idx] - self.local_oscillator_frequency) * TAU
                    / self.sample_rate;
            let end_freq_prepared = (self.y_frequencies[end_idx] - self.local_oscillator_frequency)
                * TAU
                / self.sample_rate;
            let amplitude_prepared = self.signal_amplitude / self.turn_on_time;

            let freq_diff = end_freq_prepared - start_freq_prepared;

            let atom_speed_prepared = if freq_diff > 0.0 {
                self.atom_speed * TAU / self.sample_rate
            } else {
                -self.atom_speed * TAU / self.sample_rate
            };
            let move_time = self.turn_on_time + freq_diff.abs() / atom_speed_prepared;

            Zip::indexed(&mut y_signal_i).par_for_each(|i, y| {
                let t = i as f64 / self.sample_rate;
                if t < self.turn_on_time {
                    let amplitude = amplitude_prepared * t;
                    *y += amplitude * (i as f64 * start_freq_prepared).cos();
                } else if t < move_time {
                    let freq = start_freq_prepared + atom_speed_prepared * (t - self.turn_on_time);
                    *y += self.signal_amplitude * (i as f64 * freq).cos();
                } else if t < move_time + self.turn_on_time {
                    let amplitude = amplitude_prepared * (move_time + self.turn_on_time - t);
                    *y += amplitude * (i as f64 * end_freq_prepared).cos();
                }
            });

            Zip::indexed(&mut y_signal_q).par_for_each(|i, y| {
                let t = i as f64 / self.sample_rate;
                if t < self.turn_on_time {
                    let amplitude = amplitude_prepared * t;
                    *y += amplitude * (i as f64 * start_freq_prepared).sin();
                } else if t < move_time {
                    let freq = start_freq_prepared + atom_speed_prepared * (t - self.turn_on_time);
                    *y += self.signal_amplitude * (i as f64 * freq).sin();
                } else if t < move_time + self.turn_on_time {
                    let amplitude = amplitude_prepared * (move_time + self.turn_on_time - t);
                    *y += amplitude * (i as f64 * end_freq_prepared).sin();
                }
            });
        }

        Signal {
            x_signal_i,
            x_signal_q,
            y_signal_i,
            y_signal_q,
        }
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> DebugPls for Trap<WIDTH, HEIGHT> {
    fn fmt(&self, f: Formatter) {
        f.debug_list()
            .entries(self.0.iter().map(|line| {
                line.iter()
                    .map(|&is_filled| if is_filled { 1 } else { 0 })
                    .collect::<Vec<u8>>()
            }))
            .finish()
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> DebugPls for ShiftedTrap<WIDTH, HEIGHT> {
    fn fmt(&self, f: Formatter) {
        f.debug_list()
            .entries(self.array.iter().map(|line| {
                line.iter()
                    .map(|&is_filled| if is_filled { 1 } else { 0 })
                    .collect::<Vec<u8>>()
            }))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn one_atom() {
        let trap = Trap([[true, false, false, false]]);

        let (_, shifted_trap) = Trap::shift(trap);

        let expected_trap = Trap([[false, true, false, false]]);

        assert_eq!(shifted_trap.get_trap(), expected_trap);
    }

    #[test]
    fn a_few_atoms() {
        let trap = Trap([[true, false, false, false, true, false, true, true]]);

        let (_, shifted_trap) = Trap::shift(trap);

        let expected_trap = Trap([[false, false, false, true, true, true, true, false]]);

        assert_eq!(shifted_trap.get_trap(), expected_trap);
    }

    #[test]
    fn big_array_shift() {
        let trap = Trap::from_nums(&[
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]);

        let (_, shifted_trap) = Trap::shift(trap);

        let expected_trap = Trap::from_nums(&[
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]);

        assert_eq!(shifted_trap.get_trap(), expected_trap);
    }

    #[test]
    fn big_array() {
        let trap = Trap::from_nums(&[
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]);

        let trap = ShiftedTrap::merge(Trap::shift(trap).1).1;

        let expected_trap = Trap::from_nums(&[
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);

        assert_eq!(trap, expected_trap);
    }

    #[test]
    fn big_array2() {
        let trap = Trap::from_nums(&[
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]);

        let trap = ShiftedTrap::merge(Trap::shift(trap).1).1;

        let expected_trap = Trap::from_nums(&[
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);

        assert_eq!(trap, expected_trap);
    }
    #[test]
    fn huge_array() {
        let trap = Trap::from_nums(&[
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
        ]);

        let trap = ShiftedTrap::merge(Trap::shift(trap).1).1;

        let expected_trap = Trap::from_nums(&[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]);

        assert_eq!(trap, expected_trap);
    }
}
