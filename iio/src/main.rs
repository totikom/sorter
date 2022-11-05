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
    let ctx =
        iio::Context::with_backend(iio::Backend::Network(URL)).expect("Failed to connect to board");

    let devices = vec![DEV_NAME, DEV_STREAM_TX_NAME, DEV_STREAM_RX_NAME];
    for name in devices {
        let dev = ctx.find_device(name).expect("No such device");

        println!(
            "{} [{}]: {} channel(s)",
            dev.id().unwrap_or_default(),
            dev.name().unwrap_or_default(),
            dev.num_channels(),
        );
        for channel in dev.channels() {
            println!(
                "\tid: {}, is_output: {}, type: {:?}",
                channel.id().unwrap_or_default(),
                channel.is_output(),
                channel.channel_type()
            );
            for (attr, value) in channel.attr_read_all().unwrap() {
                println!("\t\t{}:\t{}", attr, value);
            }
        }
        println!("\tDevice attributes:");
        for (attr, value) in dev.attr_read_all().unwrap() {
            println!("\t\t{}:\t{}", attr, value);
        }
    }
}
