//! # Chrono-TZ
//!
//! `Chrono-TZ` is a library that provides implementors of the
//! [`TimeZone`][timezone] trait for [`chrono`][chrono]. The
//! impls are generated by a build script using the [`IANA database`][iana]
//! and [`zoneinfo_parse`][zoneinfo_parse].
//!
//! [chrono]: https://github.com/lifthrasiir/rust-chrono
//! [timezone]: https://lifthrasiir.github.io/rust-chrono/chrono/offset/trait.TimeZone.html
//! [iana]: http://www.iana.org/time-zones
//! [zoneinfo_parse]: https://github.com/rust-datetime/zoneinfo-parse
//!
//! ## Examples
//!
//! Create a time in one timezone and convert it to UTC
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::{TimeZone, Utc};
//! use chrono_tz::US::Pacific;
//!
//! # fn main() {
//! let pacific_time = Pacific.ymd(1990, 5, 6).and_hms(12, 30, 45);
//! let utc_time = pacific_time.with_timezone(&Utc);
//! assert_eq!(utc_time, Utc.ymd(1990, 5, 6).and_hms(19, 30, 45));
//! # }
//! ```
//!
//! Create a naive datetime and convert it to a timezone-aware datetime
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::{TimeZone, NaiveDate};
//! use chrono_tz::Africa::Johannesburg;
//!
//! # fn main() {
//! let naive_dt = NaiveDate::from_ymd(2038, 1, 19).and_hms(3, 14, 08);
//! let tz_aware = Johannesburg.from_local_datetime(&naive_dt).unwrap();
//! assert_eq!(tz_aware.to_string(), "2038-01-19 03:14:08 SAST");
//! # }
//! ```
//!
//! London and New York change their clocks on different days in March
//! so only have a 4-hour difference on certain days.
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::TimeZone;
//! use chrono_tz::Europe::London;
//! use chrono_tz::America::New_York;
//!
//! # fn main() {
//! let london_time = London.ymd(2016, 3, 18).and_hms(3, 0, 0);
//! let ny_time = london_time.with_timezone(&New_York);
//! assert_eq!(ny_time, New_York.ymd(2016, 3, 17).and_hms(23, 0, 0));
//! # }
//! ```
//!
//! Adding 24 hours across a daylight savings change causes a change
//! in local time
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::{TimeZone, Duration};
//! use chrono_tz::Europe::London;
//!
//! # fn main() {
//! let dt = London.ymd(2016, 10, 29).and_hms(12, 0, 0);
//! let later = dt + Duration::hours(24);
//! assert_eq!(later, London.ymd(2016, 10, 30).and_hms(11, 0, 0));
//! # }
//! ```
//!
//! And of course you can always convert a local time to a unix timestamp
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::TimeZone;
//! use chrono_tz::Asia::Kolkata;
//!
//! # fn main() {
//! let dt = Kolkata.ymd(2000, 1, 1).and_hms(0, 0, 0);
//! let timestamp = dt.timestamp();
//! assert_eq!(timestamp, 946665000);
//! # }
//! ```
//!
//! Pretty-printing a string will use the correct abbreviation for the timezone
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::TimeZone;
//! use chrono_tz::Europe::London;
//!
//! # fn main() {
//! let dt = London.ymd(2016, 5, 10).and_hms(12, 0, 0);
//! assert_eq!(dt.to_string(), "2016-05-10 12:00:00 BST");
//! assert_eq!(dt.to_rfc3339(), "2016-05-10T12:00:00+01:00");
//! # }
//! ```
//!
//! You can convert a timezone string to a timezone using the `FromStr` trait
//!
//! ```
//! # extern crate chrono;
//! # extern crate chrono_tz;
//! use chrono::TimeZone;
//! use chrono_tz::Tz;
//! use chrono_tz::UTC;
//!
//! # fn main() {
//! let tz: Tz = "Antarctica/South_Pole".parse().unwrap();
//! let dt = tz.ymd(2016, 10, 22).and_hms(12, 0, 0);
//! let utc = dt.with_timezone(&UTC);
//! assert_eq!(utc.to_string(), "2016-10-21 23:00:00 UTC");
//! # }
//! ```
//!
//! If you need to iterate over all variants you can use the `TZ_VARIANTS` array
//! ```
//! use chrono_tz::{TZ_VARIANTS, Tz};
//! assert!(TZ_VARIANTS.iter().any(|v| *v == Tz::UTC));
//! ```

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(feature = "serde")]
mod serde;

mod binary_search;
mod directory;
mod timezone_impl;
mod timezones;

pub use crate::directory::*;
pub use crate::timezone_impl::{OffsetComponents, OffsetName};
pub use crate::timezones::ParseError;
pub use crate::timezones::Tz;
pub use crate::timezones::TZ_VARIANTS;
pub use crate::IANA_TZDB_VERSION;

#[cfg(test)]
mod tests {
    use super::America::Danmarkshavn;
    use super::Asia::Dhaka;
    use super::Australia::Adelaide;
    use super::Europe::Berlin;
    use super::Europe::London;
    use super::Europe::Moscow;
    use super::Europe::Vilnius;
    use super::Europe::Warsaw;
    use super::Pacific::Apia;
    use super::Pacific::Noumea;
    use super::Pacific::Tahiti;
    use super::Tz;
    use super::IANA_TZDB_VERSION;
    use super::US::Eastern;
    use super::UTC;
    use chrono::{Duration, NaiveDate, TimeZone};

    #[test]
    fn london_to_berlin() {
        let dt = London.with_ymd_and_hms(2016, 10, 8, 17, 0, 0).unwrap();
        let converted = dt.with_timezone(&Berlin);
        let expected = Berlin.with_ymd_and_hms(2016, 10, 8, 18, 0, 0).unwrap();
        assert_eq!(converted, expected);
    }

    #[test]
    fn us_eastern_dst_commutativity() {
        let dt = UTC.with_ymd_and_hms(2002, 4, 7, 7, 0, 0).unwrap();
        for days in -420..720 {
            let dt1 = (dt + Duration::days(days)).with_timezone(&Eastern);
            let dt2 = dt.with_timezone(&Eastern) + Duration::days(days);
            assert_eq!(dt1, dt2);
        }
    }

    #[test]
    fn test_addition_across_dst_boundary() {
        use chrono::TimeZone;
        let two_hours = Duration::hours(2);
        let edt = Eastern.with_ymd_and_hms(2019, 11, 3, 0, 0, 0).unwrap();
        let est = edt + two_hours;

        assert_eq!(edt.to_string(), "2019-11-03 00:00:00 EDT".to_string());
        assert_eq!(est.to_string(), "2019-11-03 01:00:00 EST".to_string());
        assert_eq!(est.timestamp(), edt.timestamp() + two_hours.num_seconds());
    }

    #[test]
    fn warsaw_tz_name() {
        let dt = UTC.with_ymd_and_hms(1915, 8, 4, 22, 35, 59).unwrap();
        assert_eq!(dt.with_timezone(&Warsaw).format("%Z").to_string(), "WMT");
        let dt = dt + Duration::seconds(1);
        assert_eq!(dt.with_timezone(&Warsaw).format("%Z").to_string(), "CET");
    }

    #[test]
    fn vilnius_utc_offset() {
        let dt = UTC.with_ymd_and_hms(1916, 12, 31, 22, 35, 59).unwrap().with_timezone(&Vilnius);
        assert_eq!(dt, Vilnius.with_ymd_and_hms(1916, 12, 31, 23, 59, 59).unwrap());
        let dt = dt + Duration::seconds(1);
        assert_eq!(dt, Vilnius.with_ymd_and_hms(1917, 1, 1, 0, 11, 36).unwrap());
    }

    #[test]
    fn victorian_times() {
        let dt = UTC.with_ymd_and_hms(1847, 12, 1, 0, 1, 14).unwrap().with_timezone(&London);
        assert_eq!(dt, London.with_ymd_and_hms(1847, 11, 30, 23, 59, 59).unwrap());
        let dt = dt + Duration::seconds(1);
        assert_eq!(dt, London.with_ymd_and_hms(1847, 12, 1, 0, 1, 15).unwrap());
    }

    #[test]
    fn london_dst() {
        let dt = London.with_ymd_and_hms(2016, 3, 10, 5, 0, 0).unwrap();
        let later = dt + Duration::days(180);
        let expected = London.with_ymd_and_hms(2016, 9, 6, 6, 0, 0).unwrap();
        assert_eq!(later, expected);
    }

    #[test]
    fn international_date_line_change() {
        let dt = UTC.with_ymd_and_hms(2011, 12, 30, 9, 59, 59).unwrap().with_timezone(&Apia);
        assert_eq!(dt, Apia.with_ymd_and_hms(2011, 12, 29, 23, 59, 59).unwrap());
        let dt = dt + Duration::seconds(1);
        assert_eq!(dt, Apia.with_ymd_and_hms(2011, 12, 31, 0, 0, 0).unwrap());
    }

    #[test]
    fn negative_offset_with_minutes_and_seconds() {
        let dt = UTC.with_ymd_and_hms(1900, 1, 1, 12, 0, 0).unwrap().with_timezone(&Danmarkshavn);
        assert_eq!(dt, Danmarkshavn.with_ymd_and_hms(1900, 1, 1, 10, 45, 20).unwrap());
    }

    #[test]
    fn monotonicity() {
        let mut dt = Noumea.with_ymd_and_hms(1800, 1, 1, 12, 0, 0).unwrap();
        for _ in 0..24 * 356 * 400 {
            let new = dt + Duration::hours(1);
            assert!(new > dt);
            assert!(new.with_timezone(&UTC) > dt.with_timezone(&UTC));
            dt = new;
        }
    }

    fn test_inverse<T: TimeZone>(tz: T, begin: i32, end: i32) {
        for y in begin..end {
            for d in 1..366 {
                let date = NaiveDate::from_yo_opt(y, d).unwrap();
                for h in 0..24 {
                    for m in 0..60 {
                        let dt = date.and_hms_opt(h, m, 0).unwrap().and_utc();
                        let with_tz = dt.with_timezone(&tz);
                        let utc = with_tz.with_timezone(&UTC);
                        assert_eq!(dt, utc);
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_london() {
        test_inverse(London, 1989, 1994);
    }

    #[test]
    fn inverse_dhaka() {
        test_inverse(Dhaka, 1995, 2000);
    }

    #[test]
    fn inverse_apia() {
        test_inverse(Apia, 2011, 2012);
    }

    #[test]
    fn inverse_tahiti() {
        test_inverse(Tahiti, 1911, 1914);
    }

    #[test]
    fn string_representation() {
        let dt = UTC.with_ymd_and_hms(2000, 9, 1, 12, 30, 15).unwrap().with_timezone(&Adelaide);
        assert_eq!(dt.to_string(), "2000-09-01 22:00:15 ACST");
        assert_eq!(format!("{:?}", dt), "2000-09-01T22:00:15ACST");
        assert_eq!(dt.to_rfc3339(), "2000-09-01T22:00:15+09:30");
        assert_eq!(format!("{}", dt), "2000-09-01 22:00:15 ACST");
    }

    #[test]
    fn tahiti() {
        let dt = UTC.with_ymd_and_hms(1912, 10, 1, 9, 58, 16).unwrap().with_timezone(&Tahiti);
        let before = dt - Duration::hours(1);
        assert_eq!(before, Tahiti.with_ymd_and_hms(1912, 9, 30, 23, 0, 0).unwrap());
        let after = dt + Duration::hours(1);
        assert_eq!(after, Tahiti.with_ymd_and_hms(1912, 10, 1, 0, 58, 16).unwrap());
    }

    #[test]
    fn nonexistent_time() {
        assert!(London.with_ymd_and_hms(2016, 3, 27, 1, 30, 0).single().is_none());
    }

    #[test]
    fn nonexistent_time_2() {
        assert!(London.with_ymd_and_hms(2016, 3, 27, 1, 0, 0).single().is_none());
    }

    #[test]
    fn time_exists() {
        assert!(London.with_ymd_and_hms(2016, 3, 27, 2, 0, 0).single().is_some());
    }

    #[test]
    fn ambiguous_time() {
        let ambiguous = London.with_ymd_and_hms(2016, 10, 30, 1, 0, 0);
        let earliest_utc =
            NaiveDate::from_ymd_opt(2016, 10, 30).unwrap().and_hms_opt(0, 0, 0).unwrap();
        assert_eq!(ambiguous.earliest().unwrap(), London.from_utc_datetime(&earliest_utc));
        let latest_utc =
            NaiveDate::from_ymd_opt(2016, 10, 30).unwrap().and_hms_opt(1, 0, 0).unwrap();
        assert_eq!(ambiguous.latest().unwrap(), London.from_utc_datetime(&latest_utc));
    }

    #[test]
    fn ambiguous_time_2() {
        let ambiguous = London.with_ymd_and_hms(2016, 10, 30, 1, 30, 0);
        let earliest_utc =
            NaiveDate::from_ymd_opt(2016, 10, 30).unwrap().and_hms_opt(0, 30, 0).unwrap();
        assert_eq!(ambiguous.earliest().unwrap(), London.from_utc_datetime(&earliest_utc));
        let latest_utc =
            NaiveDate::from_ymd_opt(2016, 10, 30).unwrap().and_hms_opt(1, 30, 0).unwrap();
        assert_eq!(ambiguous.latest().unwrap(), London.from_utc_datetime(&latest_utc));
    }

    #[test]
    fn ambiguous_time_3() {
        let ambiguous = Moscow.with_ymd_and_hms(2014, 10, 26, 1, 30, 0);
        let earliest_utc =
            NaiveDate::from_ymd_opt(2014, 10, 25).unwrap().and_hms_opt(21, 30, 0).unwrap();
        assert_eq!(
            ambiguous.earliest().unwrap().fixed_offset(),
            Moscow.from_utc_datetime(&earliest_utc).fixed_offset()
        );
        let latest_utc =
            NaiveDate::from_ymd_opt(2014, 10, 25).unwrap().and_hms_opt(22, 30, 0).unwrap();
        assert_eq!(ambiguous.latest().unwrap(), Moscow.from_utc_datetime(&latest_utc));
    }

    #[test]
    fn ambiguous_time_4() {
        let ambiguous = Moscow.with_ymd_and_hms(2014, 10, 26, 1, 0, 0);
        let earliest_utc =
            NaiveDate::from_ymd_opt(2014, 10, 25).unwrap().and_hms_opt(21, 0, 0).unwrap();
        assert_eq!(
            ambiguous.earliest().unwrap().fixed_offset(),
            Moscow.from_utc_datetime(&earliest_utc).fixed_offset()
        );
        let latest_utc =
            NaiveDate::from_ymd_opt(2014, 10, 25).unwrap().and_hms_opt(22, 0, 0).unwrap();
        assert_eq!(ambiguous.latest().unwrap(), Moscow.from_utc_datetime(&latest_utc));
    }

    #[test]
    fn unambiguous_time() {
        assert!(London.with_ymd_and_hms(2016, 10, 30, 2, 0, 0).single().is_some());
    }

    #[test]
    fn unambiguous_time_2() {
        assert!(Moscow.with_ymd_and_hms(2014, 10, 26, 2, 0, 0).single().is_some());
    }

    #[test]
    fn test_get_name() {
        assert_eq!(London.name(), "Europe/London");
        assert_eq!(Tz::Africa__Abidjan.name(), "Africa/Abidjan");
        assert_eq!(Tz::UTC.name(), "UTC");
        assert_eq!(Tz::Zulu.name(), "Zulu");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", London), "Europe/London");
        assert_eq!(format!("{}", Tz::Africa__Abidjan), "Africa/Abidjan");
        assert_eq!(format!("{}", Tz::UTC), "UTC");
        assert_eq!(format!("{}", Tz::Zulu), "Zulu");
    }

    #[test]
    fn test_impl_hash() {
        #[allow(dead_code)]
        #[derive(Hash)]
        struct Foo(Tz);
    }

    #[test]
    fn test_iana_tzdb_version() {
        // Format should be something like 2023c.
        assert_eq!(5, IANA_TZDB_VERSION.len());
        let numbers: Vec<&str> = IANA_TZDB_VERSION.matches(char::is_numeric).collect();
        assert_eq!(4, numbers.len());
        assert!(IANA_TZDB_VERSION.ends_with(|c: char| c.is_ascii_lowercase()));
    }
}
