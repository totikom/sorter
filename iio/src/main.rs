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
const DEV_NAME: &str = "ad7291";
fn main() {
    let ctx =
        iio::Context::with_backend(iio::Backend::Network(URL)).expect("Failed to connect to board");

    let dev = ctx.find_device(DEV_NAME).expect("No such device");

    print!("  {} ", dev.id().unwrap_or_default());
    print!("[{}]", dev.name().unwrap_or_default());
    println!(": {} channel(s)", dev.num_channels());
    for channel in dev.channels() {
        println!(
            "id: {}, is_output: {}, type: {:?}",
            channel.id().unwrap_or_default(),
            channel.is_output(),
            channel.channel_type()
        );
    }
}
