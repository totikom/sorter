use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use sorter::{ShiftedTrap, Trap, TrapParams};

fn fill_trap<R: Rng + ?Sized, const WIDTH: usize, const HEIGHT: usize>(
    rng: &mut R,
) -> Trap<WIDTH, HEIGHT> {
    let trap: [[bool; WIDTH]; HEIGHT] = rng.gen();

    Trap::from_bools(trap)
}

pub fn sorter(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("Sorter benchmark");
    group.bench_function("5x5", |b| {
        b.iter_batched(
            || fill_trap::<_, 5, 5>(&mut rng),
            |mut trap| {
                let (_, shifted_trap) = trap.shift();
                ShiftedTrap::merge(shifted_trap, &mut trap)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("10x10", |b| {
        b.iter_batched(
            || fill_trap::<_, 10, 10>(&mut rng),
            |mut trap| {
                let (_, shifted_trap) = trap.shift();
                ShiftedTrap::merge(shifted_trap, &mut trap)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}
pub fn generator(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let params = TrapParams {
        x_frequencies: [5e6, 5.2e6, 5.4e6, 5.6e6, 5.8e6], // in Hz
        y_frequencies: [5e6, 5.2e6, 5.4e6, 5.6e6, 5.8e6], // in Hz
        turn_on_time: 50e-6, // is seconds, time to turn on the laser
        local_oscillator_frequency: 100e6, // in Hz
        signal_amplitude: 1e15, // amplitude of one harmonic
        buff_size: 8192,     // Size of the SDR buffer
        sample_rate: 61.44e6, // Sample rate of the SDR
        atom_speed: 0.0175e12, // in Hz/s
    };
    let mut group = c.benchmark_group("Sorter benchmark");
    group.bench_function("5x5", |b| {
        b.iter_batched(
            || fill_trap::<_, 5, 5>(&mut rng),
            |mut trap| params.generate_sorting_signal(&mut trap),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

//pub fn generator(c: &mut Criterion) {
//let mut rng = rand::thread_rng();

//let mut group = c.benchmark_group("Sorter benchmark");
//group.bench_function("5x5", |b| {
//b.iter_batched(
//|| fill_trap::<_, 5, 5>(&mut rng),
//|mut trap| {
//let (_, shifted_trap) = trap.shift();
//ShiftedTrap::merge(shifted_trap, &mut trap)
//},
//BatchSize::SmallInput,
//);
//});
//group.bench_function("10x10", |b| {
//b.iter_batched(
//|| fill_trap::<_, 10, 10>(&mut rng),
//|mut trap| {
//let (_, shifted_trap) = trap.shift();
//ShiftedTrap::merge(shifted_trap, &mut trap)
//},
//BatchSize::SmallInput,
//);
//});
//group.finish();
//}

criterion_group!(benches, sorter, generator);
criterion_main!(benches);
