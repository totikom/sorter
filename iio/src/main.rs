// industrial-io/examples/riio_detect.rs
//
// Simple Rust IIO example to list the devices found in the specified context.
//
// Note that, if no context is requested at the command line, this will create
// a network context if the IIOD_REMOTE environment variable is set, otherwise
// it will create a local context. See Context::new().
//
// Copyright (c) 2018-2019, Frank Pagliughi
//
// Licensed under the MIT license:
//   <LICENSE or http://opensource.org/licenses/MIT>
// This file may not be copied, modified, or distributed except according
// to those terms.
//
use industrial_io as iio;
use plotters::prelude::*;
use std::process;

const URL: &str = "172.16.1.246";
const DEV_NAME: &str = "ad9361-phy";
const DEV_STREAM_TX_NAME: &str = "cf-ad9361-dds-core-lpc";
const DEV_STREAM_RX_NAME: &str = "cf-ad9361-lpc";
const OUT_FILE_NAME: &'static str = "target/sample.png";

fn main() {
    println!("* Acquiring IIO context");
    let ctx =
        iio::Context::with_backend(iio::Backend::Network(URL)).expect("Failed to connect to board");

    //let devices = vec![DEV_NAME, DEV_STREAM_TX_NAME, DEV_STREAM_RX_NAME];
    //for name in devices {
    //let dev = ctx.find_device(name).expect("No such device");

    //println!(
    //"{} [{}]: {} channel(s)",
    //dev.id().unwrap_or_default(),
    //dev.name().unwrap_or_default(),
    //dev.num_channels(),
    //);
    //for channel in dev.channels() {
    //println!(
    //"\tid: {}, is_output: {}, type: {:?}",
    //channel.id().unwrap_or_default(),
    //channel.is_output(),
    //channel.channel_type()
    //);
    //for (attr, value) in channel.attr_read_all().unwrap() {
    //println!("\t\t{}:\t{}", attr, value);
    //}
    //}
    //println!("\tDevice attributes:");
    //for (attr, value) in dev.attr_read_all().unwrap() {
    //println!("\t\t{}:\t{}", attr, value);
    //}
    //}

    let rx_cfg = StreamCfg {
        bandwidth: 2_000_000,
        samplerate: 2_500_000,
        local_oscillator: 2_500_000_000,
        port: "A_BALANCED".to_string(),
    };

    let tx_cfg = StreamCfg {
        bandwidth: 1_500_000,
        samplerate: 2_500_000,
        local_oscillator: 2_500_000_000,
        port: "A".to_string(),
    };

    println!("* Acquiring AD9361 streaming devices");
    let phy_dev = ctx.find_device(DEV_NAME).expect("No phy device");
    let tx_dev = ctx.find_device(DEV_STREAM_TX_NAME).expect("No TX device");
    let rx_dev = ctx.find_device(DEV_STREAM_RX_NAME).expect("No RX device");

    println!("* Acquiring AD9361 channels");
    let phy_chn_rx = phy_dev
        .find_channel("voltage0", false)
        .expect("No voltage0 input channel");
    let phy_chn_tx = phy_dev
        .find_channel("voltage0", true)
        .expect("No voltage0 input channel");

    let phy_lo_chn_rx = phy_dev
        .find_channel("altvoltage0", true)
        .expect("No altvoltage0 input channel");
    let phy_lo_chn_tx = phy_dev
        .find_channel("altvoltage1", true)
        .expect("No altvoltage1 input channel");

    println!("* Configuring AD9361 for streaming");
    phy_chn_rx
        .attr_write_str("rf_port_select", &rx_cfg.port)
        .unwrap();
    phy_chn_rx
        .attr_write_int("rf_bandwidth", rx_cfg.bandwidth)
        .unwrap();
    phy_chn_rx
        .attr_write_int("sampling_frequency", rx_cfg.samplerate)
        .unwrap();

    phy_chn_tx
        .attr_write_str("rf_port_select", &tx_cfg.port)
        .unwrap();
    phy_chn_tx
        .attr_write_int("rf_bandwidth", tx_cfg.bandwidth)
        .unwrap();
    phy_chn_tx
        .attr_write_int("sampling_frequency", tx_cfg.samplerate)
        .unwrap();

    phy_lo_chn_rx
        .attr_write_int("frequency", rx_cfg.local_oscillator)
        .unwrap();
    phy_lo_chn_tx
        .attr_write_int("frequency", tx_cfg.local_oscillator)
        .unwrap();

    println!("* Initializing AD9361 for streaming");
    let rx_0i = rx_dev
        .find_channel("voltage0", false)
        .expect("No voltage0 input channel");
    let rx_0q = rx_dev
        .find_channel("voltage1", false)
        .expect("No voltage1 input channel");

    let tx_0i = tx_dev
        .find_channel("voltage0", true)
        .expect("No voltage0 input channel");
    let tx_0q = tx_dev
        .find_channel("voltage1", true)
        .expect("No voltage1 input channel");

    println!("* Enabling IIO streaming channels");
    rx_0i.enable();
    rx_0q.enable();
    tx_0i.enable();
    tx_0q.enable();

    println!("* Creating non-cyclic IIO buffers with 1MiS");
    let mut rx_buf = rx_dev.create_buffer(1012 * 1024, false).unwrap();
    let tx_buf = tx_dev.create_buffer(1012 * 1024, true).unwrap();

    let single_tone_i: Vec<i16> = vec![10; 1024];
    let single_tone_q: Vec<i16> = vec![0; 1024];

    tx_0i.write(&tx_buf, &single_tone_i).unwrap();
    tx_0q.write(&tx_buf, &single_tone_q).unwrap();

    let nbytes_tx = tx_buf.push().unwrap();

    println!("Bytes pushed:{}", nbytes_tx);

    let nbytes_rx = rx_buf.refill().unwrap();
    println!("Bytes received:{}", nbytes_rx);

    let rx_i_buf: Vec<i16> = rx_0i.read(&rx_buf).unwrap();
    let rx_q_buf: Vec<i16> = rx_0q.read(&rx_buf).unwrap();

    plot(&rx_i_buf, &rx_q_buf, rx_cfg.samplerate as usize).unwrap();

    println!("Ok!");
}

fn plot(
    i_signal: &[i16],
    q_signal: &[i16],
    samplerate: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Received signal", ("sans-serif", 60))?;

    let t = 1.0 / samplerate as f64;
    let x_axis = (0.0..t * i_signal.len() as f64).step(t);

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(
            0.0f64..t * i_signal.len() as f64 / 50.0,
            -800.0f64..800.0f64,
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
        &RED,
    ))?
    .label("I-signal")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    cc.draw_series(LineSeries::new(
        x_axis.values().zip(q_signal.iter().map(|&x| f64::from(x))),
        &BLUE,
    ))?
    .label("Q-signal")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    cc.configure_series_labels().border_style(&BLACK).draw()?;

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
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}

/// common RX and TX streaming params
struct StreamCfg {
    /// Analog banwidth in Hz
    bandwidth: i64,
    /// Analog banwidth in Hz
    samplerate: i64,
    /// Local oscillator frequency in Hz
    local_oscillator: i64,
    /// Port name
    port: String,
}
