use ndarray::Array2;
fn main() {
    println!("Hello, world!");
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct TrapParams<const WIDTH: usize, const HEIGHT: usize> {
    x_frequencies: [f64; WIDTH],
    y_frequencies: [f64; HEIGHT],
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
            acc + line.iter().fold(0, |acc, x| if *x { acc } else { acc + 1 })
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
                let mut sum = line.iter().fold(0, |acc, x| if *x { acc } else { acc + 1 });

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
                            break;
                        }
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

                if !breaked {
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
}
