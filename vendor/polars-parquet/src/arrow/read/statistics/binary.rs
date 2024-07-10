use arrow::array::{MutableArray, MutableBinaryArray};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use crate::parquet::statistics::{BinaryStatistics, Statistics as ParquetStatistics};

pub(super) fn push<O: Offset>(
    from: Option<&dyn ParquetStatistics>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
) -> PolarsResult<()> {
    let min = min
        .as_mut_any()
        .downcast_mut::<MutableBinaryArray<O>>()
        .unwrap();
    let max = max
        .as_mut_any()
        .downcast_mut::<MutableBinaryArray<O>>()
        .unwrap();
    let from = from.map(|s| s.as_any().downcast_ref::<BinaryStatistics>().unwrap());
    min.push(from.and_then(|s| s.min_value.as_ref()));
    max.push(from.and_then(|s| s.max_value.as_ref()));
    Ok(())
}
