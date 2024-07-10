# SXD-XPath

An XML XPath library in Rust.

[![Linux Build Status](https://travis-ci.org/shepmaster/sxd-xpath.svg?branch=master)](https://travis-ci.org/shepmaster/sxd-xpath)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/2hrixanb82kcaci1/branch/master?svg=true)](https://ci.appveyor.com/project/shepmaster/sxd-xpath/branch/master)
[![Current Version](http://meritbadge.herokuapp.com/sxd-xpath)](https://crates.io/crates/sxd-xpath)
[![Documentation](https://docs.rs/sxd-xpath/badge.svg)](https://docs.rs/sxd-xpath/)

## Overview

The project is broken into two crates:

1. [`document`][sxd-document] - Basic DOM manipulation and reading/writing XML from strings.
2. `xpath` - Implementation of XPath 1.0 expressions.

There are also scattered utilities for playing around at the command
line.

In the future, I hope to add support for XSLT 1.0.

[sxd-document]: https://github.com/shepmaster/sxd-document/

## Goals

This project has a lofty goal: replace [libxml] and [libxslt].

[libxml]: http://xmlsoft.org/
[libxslt]: http://xmlsoft.org/

## Contributing

1. Fork it ( https://github.com/shepmaster/sxd-xpath/fork )
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Add a failing test.
4. Add code to pass the test.
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Ensure tests pass.
7. Push to the branch (`git push origin my-new-feature`)
8. Create a new Pull Request

## License

Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
