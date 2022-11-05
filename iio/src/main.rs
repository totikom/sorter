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
use std::process;

const URL: &str = "172.16.1.246";
const DEV_NAME: &str = "ad9361-phy";
const DEV_STREAM_TX_NAME: &str = "cf-ad9361-dds-core-lpc";
const DEV_STREAM_RX_NAME: &str = "cf-ad9361-lpc";

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

    dbg!(rx_0i.data_format());
    dbg!(rx_0q.data_format());
    dbg!(tx_0i.data_format());
    dbg!(tx_0q.data_format());

    println!("* Creating non-cyclic IIO buffers with 1MiS");
    let mut rx_buf = rx_dev.create_buffer(1012 * 1024, false).unwrap();
    let mut tx_buf = tx_dev.create_buffer(1012 * 1024, false).unwrap();

    // RX and TX sample counters
    let mut nrx = 0;
    let mut ntx = 0;

    let rx_length = 2 * rx_0i.data_format().length() as usize;
    let tx_length = 2 * tx_0i.data_format().length() as usize;
    loop {
        let nbytes_tx = tx_buf.push().unwrap();
        let nbytes_rx = rx_buf.refill().unwrap();

        let rx_i_buf: Vec<i16> = rx_0i.read(&rx_buf).unwrap();
        let rx_q_buf: Vec<i16> = rx_0q.read(&rx_buf).unwrap();

        // Example: swap I and Q components

        tx_0i.write(&tx_buf, &rx_q_buf).unwrap();
        tx_0q.write(&tx_buf, &rx_i_buf).unwrap();

        nrx += nbytes_rx / rx_length;
        ntx += nbytes_tx / tx_length;
        println!(
            "\tRX {:8.2} MSmp, TX {:8.2} MSmp",
            nrx as f32 / 1e6,
            ntx as f32 / 1e6
        );
    }
    println!("Ok!");
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
