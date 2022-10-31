use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use sorter::{ShiftedTrap, Trap};

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

criterion_group!(benches, sorter);
criterion_main!(benches);
