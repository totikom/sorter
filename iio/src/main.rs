use ad9361_iio as ad9361;
use industrial_io as iio;

use ad9361::{RxPortSelect, Signal, TxPortSelect, AD9361};
use log::info;
use plotters::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::{FRAC_PI_2, TAU};

const URL: &str = "172.16.1.246";
const OUT_FILE_NAME_0: &str = "target/sample_0.png";
const OUT_FILE_NAME_1: &str = "target/sample_1.png";
const SPECTRUM_FILE_NAME_0: &str = "target/spectrum_0.png";
const SPECTRUM_FILE_NAME_1: &str = "target/spectrum_1.png";

fn main() {
    env_logger::init();

    info!("acquiring IIO context");
    let ctx =
        iio::Context::with_backend(iio::Backend::Network(URL)).expect("Failed to connect to board");

    info!("acquiring AD9361");
    let ad = AD9361::from_ctx(&ctx).expect("Failed to acquire AD9361");

    let mut rx = ad.rx.borrow_mut();
    let mut tx = ad.tx.borrow_mut();

    dbg!(rx.hardware_gain(0).unwrap());
    dbg!(rx.hardware_gain(1).unwrap());
    dbg!(tx.hardware_gain(0).unwrap());
    dbg!(tx.hardware_gain(1).unwrap());

    let rx_cfg = RxStreamCfg {
        bandwidth: 12_800_000,
        samplerate: 16_000_000,
        local_oscillator: 99_000_000,
        port: RxPortSelect::BBalanced,
    };

    let tx_cfg = TxStreamCfg {
        bandwidth: 12_800_000,
        samplerate: 16_000_000,
        local_oscillator: 99_000_000,
        port: TxPortSelect::A,
    };

    info!("configuring local oscillators");
    rx.set_lo(rx_cfg.local_oscillator).unwrap();
    tx.set_lo(tx_cfg.local_oscillator).unwrap();

    info!("configuring receiver channel 0");
    rx.set_rf_bandwidth(0, rx_cfg.bandwidth).unwrap();
    rx.set_sampling_frequency(0, rx_cfg.samplerate).unwrap();
    rx.set_port(0, rx_cfg.port).unwrap();

    info!("configuring receiver channel 1");
    rx.set_rf_bandwidth(1, rx_cfg.bandwidth).unwrap();
    rx.set_sampling_frequency(1, rx_cfg.samplerate).unwrap();
    rx.set_port(1, RxPortSelect::ABalanced).unwrap();

    info!("configuring transmitter channel 0");
    tx.set_rf_bandwidth(0, tx_cfg.bandwidth).unwrap();
    tx.set_sampling_frequency(0, tx_cfg.samplerate).unwrap();
    tx.set_port(0, TxPortSelect::A).unwrap();

    info!("enabling channels");
    rx.enable(0);
    rx.enable(1);
    tx.enable(0);
    //tx.enable(1);

    info!("creating non-cyclic IIO buffers");
    tx.create_buffer(8192, true).unwrap();
    rx.create_buffer(8192, false).unwrap();

    let params_sin = SignalParams {
        harmonics: vec![
            Harmonic {
                amplitude: 10_000.0,
                frequency: 1_100_000.0,
                phase: 0.0,
            },
            Harmonic {
                amplitude: 10_000.0,
                frequency: -1_300_000.0,
                phase: 0.0,
            },
        ],
        len: 8192,
        samplerate: tx_cfg.samplerate as usize,
    };

    let params_cos = SignalParams {
        harmonics: vec![
            Harmonic {
                amplitude: 10_000.0,
                frequency: 1_100_000.0,
                phase: FRAC_PI_2,
            },
            Harmonic {
                amplitude: 10_000.0,
                frequency: -1_300_000.0,
                phase: FRAC_PI_2,
            },
        ],
        len: 8192,
        samplerate: tx_cfg.samplerate as usize,
    };

    let signal_0 = Signal {
        i_channel: generate_signal(&params_sin),
        q_channel: generate_signal(&params_cos),
    };

    let signal_1 = Signal {
        i_channel: vec![0; 8192],
        q_channel: vec![0; 8192],
    };

    let (i_count, q_count) = tx.write(0, &signal_0).unwrap();
    info!(
        "written {} and {} samples to buffer of the channel 0",
        i_count, q_count
    );

    //let (i_count, q_count) = tx.write(1, &signal_1).unwrap();
    //info!(
    //"written {} and {} bytes to buffer of the channel 1",
    //i_count, q_count
    //);

    let bytes_pushed = tx.push_samples_to_device().unwrap();
    info!("send {} bytes to device", bytes_pushed);

    let bytes_pooled = rx.pool_samples_to_buff().unwrap();
    info!("received {} bytes from device", bytes_pooled);

    let signal_received_0 = rx.read(0).unwrap();
    let signal_received_1 = rx.read(1).unwrap();
    dbg!(signal_received_0.i_channel.len());
    dbg!(signal_received_1.i_channel.len());

    info!("plotting graphs");
    plot(
        &signal_received_0,
        rx_cfg.samplerate as usize,
        OUT_FILE_NAME_0,
        100.0,
    )
    .unwrap();

    plot(&signal_0, rx_cfg.samplerate as usize, OUT_FILE_NAME_1, 1.0).unwrap();

    plot(
        &signal_0,
        rx_cfg.samplerate as usize,
        "target/expected_signal.png",
        100.0,
    )
    .unwrap();

    info!("generating spectra");
    let expected_spectrum_0 = spectrum(&signal_0);
    let expected_spectrum_1 = spectrum(&signal_1);
    let spectrum_0 = spectrum(&signal_received_0);

    let spectrum_1 = spectrum(&signal_received_1);

    info!("plotting spectra");
    plot_spectrum(
        &spectrum_0,
        &expected_spectrum_0,
        rx_cfg.samplerate as usize,
        SPECTRUM_FILE_NAME_0,
        rx_cfg.local_oscillator as f32,
    )
    .unwrap();

    plot_spectrum(
        &spectrum_1,
        &expected_spectrum_0,
        rx_cfg.samplerate as usize,
        SPECTRUM_FILE_NAME_1,
        rx_cfg.local_oscillator as f32,
    )
    .unwrap();

    info!("Cleaning up");
    rx.destroy_buffer();
    tx.destroy_buffer();

    rx.disable(0);
    rx.disable(1);
    tx.disable(0);
    tx.disable(1);

    info!("Finished!");
}

fn plot(
    signal: &Signal,
    samplerate: usize,
    filename: &str,
    scale: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();

    let i_signal = &signal.i_channel;
    let q_signal = &signal.q_channel;

    let min_i = *i_signal.iter().min().unwrap();
    let min_q = *q_signal.iter().min().unwrap();
    let min = std::cmp::min(min_i, min_q);

    let max_i = *i_signal.iter().max().unwrap();
    let max_q = *q_signal.iter().max().unwrap();
    let max = std::cmp::max(max_i, max_q);

    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Received signal", ("sans-serif", 60))?;

    let t = 1.0 / samplerate as f64;
    let x_axis = (0.0..t * i_signal.len() as f64).step(t);

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(
            0.0f64..t * i_signal.len() as f64 / scale,
            f64::from(min)..f64::from(max),
        )?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        //.disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v * 1000.0))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
        x_axis.values().zip(i_signal.iter().map(|&x| f64::from(x))),
        RED,
    ))?
    .label("I-signal")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    cc.draw_series(LineSeries::new(
        x_axis.values().zip(q_signal.iter().map(|&x| f64::from(x))),
        BLUE,
    ))?
    .label("Q-signal")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    cc.configure_series_labels().border_style(BLACK).draw()?;

    /*
    // It's possible to use a existing pointing element
     cc.draw_series(PointSeries::<_, _, Circle<_>>::new(
        (-3.0f32..2.1f32).step(1.0).values().map(|x| (x, x.sin())),
        5,
        Into::<ShapeStyle>::into(&RGBColor(255,0,0)).filled(),
    ))?;*/

    // Otherwise you can use a function to construct your pointing element yourself
    //cc.draw_series(PointSeries::of_element(
    //(-3.0f32..2.1f32).step(1.0).values().map(|x| (x, x.sin())),
    //5,
    //ShapeStyle::from(&RED).filled(),
    //&|coord, size, style| {
    //EmptyElement::at(coord)
    //+ Circle::new((0, 0), size, style)
    //+ Text::new(format!("{:?}", coord), (0, 15), ("sans-serif", 15))
    //},
    //))?;

    //let drawing_areas = lower.split_evenly((1, 2));

    //for (drawing_area, idx) in drawing_areas.iter().zip(1..) {
    //let mut cc = ChartBuilder::on(&drawing_area)
    //.x_label_area_size(30)
    //.y_label_area_size(30)
    //.margin_right(20)
    //.caption(format!("y = x^{}", 1 + 2 * idx), ("sans-serif", 40))
    //.build_cartesian_2d(-1f32..1f32, -1f32..1f32)?;
    //cc.configure_mesh()
    //.x_labels(5)
    //.y_labels(3)
    //.max_light_lines(4)
    //.draw()?;

    //cc.draw_series(LineSeries::new(
    //(-1f32..1f32)
    //.step(0.01)
    //.values()
    //.map(|x| (x, x.powf(idx as f32 * 2.0 + 1.0))),
    //&BLUE,
    //))?;
    //}

    // To avoid the IO failure being ignored silently, we manually call the present function
    root_area.present().expect("Unable to write result to file");
    info!("result has been saved to {}", filename);
    Ok(())
}

fn plot_spectrum(
    signal: &Vec<Complex<f32>>,
    expected_signal: &Vec<Complex<f32>>,
    samplerate: usize,
    filename: &str,
    lo: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();

    let signal: Vec<_> = signal
        .iter()
        .map(|x| x.norm().powi(2) / (signal.len() as f32).sqrt())
        .collect();
    let expected_signal: Vec<_> = expected_signal
        .iter()
        .map(|x| x.norm().powi(2) / (expected_signal.len() as f32).sqrt())
        .collect();

    let min = signal.iter().copied().reduce(f32::min).unwrap();
    let expected_min = expected_signal.iter().copied().reduce(f32::min).unwrap();
    let min = f32::min(min, expected_min);
    let min = if min == 0.0 { 1.0 } else { min };

    let max = signal.iter().copied().reduce(f32::max).unwrap();
    let expected_max = expected_signal.iter().copied().reduce(f32::max).unwrap();
    let max = f32::max(max, expected_max);

    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Received signal", ("sans-serif", 60))?;

    let x_axis = fft_freq(signal.len() as isize, 1.0 / samplerate as f32, lo);
    let (x_min, x_max) = x_axis
        .iter()
        .copied()
        .fold((f32::MAX, f32::MIN), |(prev_min, prev_max), y| {
            (f32::min(prev_min, y), f32::max(prev_max, y))
        });

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, (min..max).log_scale())?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        //.disable_mesh()
        .x_label_formatter(&|v| format!("{:.3e}", v))
        .y_label_formatter(&|v| format!("{:.1e}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
        x_axis.iter().zip(signal.iter()).map(|(x, y)| (*x, *y)),
        RED,
    ))?
    .label("Received spectrum")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    cc.draw_series(LineSeries::new(
        x_axis
            .iter()
            .zip(expected_signal.iter())
            .map(|(x, y)| (*x, *y)),
        BLUE,
    ))?
    .label("Expected spectrum")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    cc.configure_series_labels().border_style(BLACK).draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root_area.present().expect("Unable to write result to file");
    info!("result has been saved to {}", filename);
    Ok(())
}

fn fft_freq(n: isize, d: f32, lo: f32) -> Vec<f32> {
    if n % 2 == 0 {
        (0..n / 2)
            .into_iter()
            .chain(((-n / 2)..0).into_iter())
            .map(|x| (x as f32) / (d * n as f32) + lo)
            .collect::<Vec<_>>()
    } else {
        (0..=(n - 1) / 2)
            .into_iter()
            .chain(((-(n - 1) / 2)..0).into_iter())
            .map(|x| (x as f32) / (d * n as f32) + lo)
            .collect::<Vec<_>>()
    }
}

fn fft_freq_shifted(n: isize, d: f32, lo: f32) -> Vec<f32> {
    if n % 2 == 0 {
        ((-n / 2)..n / 2)
            .into_iter()
            .map(|x| (x as f32) / (d * n as f32) + lo)
            .collect::<Vec<_>>()
    } else {
        ((-(n - 1) / 2)..=(n - 1) / 2)
            .into_iter()
            .map(|x| (x as f32) / (d * n as f32) + lo)
            .collect::<Vec<_>>()
    }
}

fn spectrum(signal: &Signal) -> Vec<Complex<f32>> {
    let mut buffer: Vec<_> = signal
        .i_channel
        .iter()
        .zip(signal.q_channel.iter())
        .map(|(&i, &q)| Complex::new(f32::from(i), f32::from(q)))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());

    fft.process(&mut buffer);
    buffer
}

/// TX streaming params
struct TxStreamCfg {
    /// Analog banwidth in Hz
    bandwidth: i64,
    /// Analog banwidth in Hz
    samplerate: i64,
    /// Local oscillator frequency in Hz
    local_oscillator: i64,
    /// Port name
    port: TxPortSelect,
}
/// RX streaming params
struct RxStreamCfg {
    /// Analog banwidth in Hz
    bandwidth: i64,
    /// Analog banwidth in Hz
    samplerate: i64,
    /// Local oscillator frequency in Hz
    local_oscillator: i64,
    /// Port name
    port: RxPortSelect,
}

#[derive(Debug)]
struct SignalParams {
    harmonics: Vec<Harmonic>,
    len: usize,
    samplerate: usize,
}

#[derive(Debug)]
struct Harmonic {
    frequency: f64,
    amplitude: f64,
    phase: f64,
}

fn generate_signal(params: &SignalParams) -> Vec<i16> {
    let mut signal = vec![0; params.len];
    for harmonic in &params.harmonics {
        for (idx, value) in signal.iter_mut().enumerate() {
            *value += (((idx as f64 / params.samplerate as f64) * TAU * harmonic.frequency
                + harmonic.phase)
                .cos()
                * harmonic.amplitude) as i16;
        }
    }
    signal
}
