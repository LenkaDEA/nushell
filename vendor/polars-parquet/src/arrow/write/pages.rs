use std::fmt::Debug;

use arrow::array::{Array, FixedSizeListArray, ListArray, MapArray, StructArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::PhysicalType;
use arrow::offset::{Offset, OffsetsBuffer};
use polars_error::{polars_bail, PolarsResult};

use super::{array_to_pages, Encoding, WriteOptions};
use crate::arrow::read::schema::is_nullable;
use crate::parquet::page::Page;
use crate::parquet::schema::types::{ParquetType, PrimitiveType as ParquetPrimitiveType};
use crate::write::DynIter;

#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveNested {
    pub is_optional: bool,
    pub validity: Option<Bitmap>,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ListNested<O: Offset> {
    pub is_optional: bool,
    pub offsets: OffsetsBuffer<O>,
    pub validity: Option<Bitmap>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedSizeListNested {
    pub validity: Option<Bitmap>,
    pub is_optional: bool,
    pub width: usize,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructNested {
    pub is_optional: bool,
    pub validity: Option<Bitmap>,
    pub length: usize,
}

impl<O: Offset> ListNested<O> {
    pub fn new(offsets: OffsetsBuffer<O>, validity: Option<Bitmap>, is_optional: bool) -> Self {
        Self {
            is_optional,
            offsets,
            validity,
        }
    }
}

/// Descriptor of nested information of a field
#[derive(Debug, Clone, PartialEq)]
pub enum Nested {
    /// a primitive (leaf or parquet column)
    Primitive(PrimitiveNested),
    List(ListNested<i32>),
    LargeList(ListNested<i64>),
    FixedSizeList(FixedSizeListNested),
    Struct(StructNested),
}

impl Nested {
    /// Returns the length (number of rows) of the element
    pub fn len(&self) -> usize {
        match self {
            Nested::Primitive(nested) => nested.length,
            Nested::List(nested) => nested.offsets.len_proxy(),
            Nested::LargeList(nested) => nested.offsets.len_proxy(),
            Nested::FixedSizeList(nested) => nested.length,
            Nested::Struct(nested) => nested.length,
        }
    }

    pub fn primitive(validity: Option<Bitmap>, is_optional: bool, length: usize) -> Self {
        Self::Primitive(PrimitiveNested {
            validity,
            is_optional,
            length,
        })
    }

    pub fn list(validity: Option<Bitmap>, is_optional: bool, offsets: OffsetsBuffer<i32>) -> Self {
        Self::List(ListNested {
            validity,
            is_optional,
            offsets,
        })
    }

    pub fn large_list(
        validity: Option<Bitmap>,
        is_optional: bool,
        offsets: OffsetsBuffer<i64>,
    ) -> Self {
        Self::LargeList(ListNested {
            validity,
            is_optional,
            offsets,
        })
    }

    pub fn fixed_size_list(
        validity: Option<Bitmap>,
        is_optional: bool,
        width: usize,
        length: usize,
    ) -> Self {
        Self::FixedSizeList(FixedSizeListNested {
            validity,
            is_optional,
            width,
            length,
        })
    }

    pub fn structure(validity: Option<Bitmap>, is_optional: bool, length: usize) -> Self {
        Self::Struct(StructNested {
            validity,
            is_optional,
            length,
        })
    }
}

/// Constructs the necessary `Vec<Vec<Nested>>` to write the rep and def levels of `array` to parquet
pub fn to_nested(array: &dyn Array, type_: &ParquetType) -> PolarsResult<Vec<Vec<Nested>>> {
    let mut nested = vec![];

    to_nested_recursive(array, type_, &mut nested, vec![])?;
    Ok(nested)
}

fn to_nested_recursive(
    array: &dyn Array,
    type_: &ParquetType,
    nested: &mut Vec<Vec<Nested>>,
    mut parents: Vec<Nested>,
) -> PolarsResult<()> {
    let is_optional = is_nullable(type_.get_field_info());

    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let fields = if let ParquetType::GroupType { fields, .. } = type_ {
                fields
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a struct array".to_string(),
                )
            };

            parents.push(Nested::Struct(StructNested {
                is_optional,
                validity: array.validity().cloned(),
                length: array.len(),
            }));

            for (type_, array) in fields.iter().zip(array.values()) {
                to_nested_recursive(array.as_ref(), type_, nested, parents.clone())?;
            }
        },
        FixedSizeList => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array".to_string(),
                )
            };

            parents.push(Nested::FixedSizeList(FixedSizeListNested {
                validity: array.validity().cloned(),
                length: array.len(),
                width: array.size(),
                is_optional,
            }));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array".to_string(),
                )
            };

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array".to_string(),
                )
            };

            parents.push(Nested::LargeList(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a map array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a map array".to_string(),
                )
            };

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.field().as_ref(), type_, nested, parents)?;
        },
        _ => {
            parents.push(Nested::Primitive(PrimitiveNested {
                validity: array.validity().cloned(),
                is_optional,
                length: array.len(),
            }));
            nested.push(parents)
        },
    }
    Ok(())
}

/// Convert [`Array`] to `Vec<&dyn Array>` leaves in DFS order.
pub fn to_leaves(array: &dyn Array) -> Vec<&dyn Array> {
    let mut leaves = vec![];
    to_leaves_recursive(array, &mut leaves);
    leaves
}

fn to_leaves_recursive<'a>(array: &'a dyn Array, leaves: &mut Vec<&'a dyn Array>) {
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            array
                .values()
                .iter()
                .for_each(|a| to_leaves_recursive(a.as_ref(), leaves));
        },
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            to_leaves_recursive(array.values().as_ref(), leaves);
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            to_leaves_recursive(array.values().as_ref(), leaves);
        },
        FixedSizeList => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            to_leaves_recursive(array.values().as_ref(), leaves);
        },
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();
            to_leaves_recursive(array.field().as_ref(), leaves);
        },
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | LargeUtf8 | Dictionary(_) | BinaryView | Utf8View => leaves.push(array),
        other => todo!("Writing {:?} to parquet not yet implemented", other),
    }
}

/// Convert `ParquetType` to `Vec<ParquetPrimitiveType>` leaves in DFS order.
pub fn to_parquet_leaves(type_: ParquetType) -> Vec<ParquetPrimitiveType> {
    let mut leaves = vec![];
    to_parquet_leaves_recursive(type_, &mut leaves);
    leaves
}

fn to_parquet_leaves_recursive(type_: ParquetType, leaves: &mut Vec<ParquetPrimitiveType>) {
    match type_ {
        ParquetType::PrimitiveType(primitive) => leaves.push(primitive),
        ParquetType::GroupType { fields, .. } => {
            fields
                .into_iter()
                .for_each(|type_| to_parquet_leaves_recursive(type_, leaves));
        },
    }
}

/// Returns a vector of iterators of [`Page`], one per leaf column in the array
pub fn array_to_columns<A: AsRef<dyn Array> + Send + Sync>(
    array: A,
    type_: ParquetType,
    options: WriteOptions,
    encoding: &[Encoding],
) -> PolarsResult<Vec<DynIter<'static, PolarsResult<Page>>>> {
    let array = array.as_ref();
    let nested = to_nested(array, &type_)?;

    let types = to_parquet_leaves(type_);

    let values = to_leaves(array);

    assert_eq!(encoding.len(), types.len());

    values
        .iter()
        .zip(nested)
        .zip(types)
        .zip(encoding.iter())
        .map(|(((values, nested), type_), encoding)| {
            array_to_pages(*values, type_, &nested, options, *encoding)
        })
        .collect()
}

pub fn arrays_to_columns<A: AsRef<dyn Array> + Send + Sync>(
    arrays: &[A],
    type_: ParquetType,
    options: WriteOptions,
    encoding: &[Encoding],
) -> PolarsResult<Vec<DynIter<'static, PolarsResult<Page>>>> {
    let array = arrays[0].as_ref();
    let nested = to_nested(array, &type_)?;

    let types = to_parquet_leaves(type_);

    // leaves; index level is nesting depth.
    // index i: has a vec because we have multiple chunks.
    let mut leaves = vec![];

    // Ensure we transpose the leaves. So that all the leaves from the same columns are at the same level vec.
    let mut scratch = vec![];
    for arr in arrays {
        scratch.clear();
        to_leaves_recursive(arr.as_ref(), &mut scratch);
        for (i, leave) in scratch.iter().copied().enumerate() {
            while i < leaves.len() {
                leaves.push(vec![]);
            }
            leaves[i].push(leave);
        }
    }

    leaves
        .into_iter()
        .zip(nested)
        .zip(types)
        .zip(encoding.iter())
        .map(move |(((values, nested), type_), encoding)| {
            let iter = values.into_iter().map(|leave_values| {
                array_to_pages(leave_values, type_.clone(), &nested, options, *encoding)
            });

            // Need a scratch to bubble up the error :/
            let mut scratch = Vec::with_capacity(iter.size_hint().0);
            for v in iter {
                scratch.push(v?)
            }
            Ok(DynIter::new(scratch.into_iter().flatten()))
        })
        .collect::<PolarsResult<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use arrow::array::*;
    use arrow::datatypes::*;

    use super::super::{FieldInfo, ParquetPhysicalType};
    use super::*;
    use crate::parquet::schema::types::{
        GroupLogicalType, PrimitiveConvertedType, PrimitiveLogicalType,
    };
    use crate::parquet::schema::Repetition;

    #[test]
    fn test_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", ArrowDataType::Boolean, false),
            Field::new("c", ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };
        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                vec![
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_struct_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", ArrowDataType::Boolean, false),
            Field::new("c", ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let fields = vec![
            Field::new("b", array.data_type().clone(), true),
            Field::new("c", array.data_type().clone(), true),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            vec![Box::new(array.clone()), Box::new(array)],
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![type_.clone(), type_],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                // a.b.b
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.b.c
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.c.b
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                // a.c.c
                vec![
                    Nested::structure(None, false, 4),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_list_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", ArrowDataType::Boolean, false),
            Field::new("c", ArrowDataType::Int32, false),
        ];

        let array = StructArray::new(
            ArrowDataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let array = ListArray::new(
            ArrowDataType::List(Box::new(Field::new("l", array.data_type().clone(), true))),
            vec![0i32, 2, 4].try_into().unwrap(),
            Box::new(array),
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "l".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "list".to_string(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![type_],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_map() {
        let kv_type = ArrowDataType::Struct(vec![
            Field::new("k", ArrowDataType::Utf8, false),
            Field::new("v", ArrowDataType::Int32, false),
        ]);
        let kv_field = Field::new("kv", kv_type.clone(), false);
        let map_type = ArrowDataType::Map(Box::new(kv_field), false);

        let key_array = Utf8Array::<i32>::from_slice(["k1", "k2", "k3", "k4", "k5", "k6"]).boxed();
        let val_array = Int32Array::from_slice([42, 28, 19, 31, 21, 17]).boxed();
        let kv_array = StructArray::try_new(kv_type, vec![key_array, val_array], None)
            .unwrap()
            .boxed();
        let offsets = OffsetsBuffer::try_from(vec![0, 2, 3, 4, 6]).unwrap();

        let array = MapArray::try_new(map_type, offsets, kv_array, None).unwrap();

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "kv".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "k".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: Some(PrimitiveLogicalType::String),
                    converted_type: Some(PrimitiveConvertedType::Utf8),
                    physical_type: ParquetPhysicalType::ByteArray,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "v".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "m".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: Some(GroupLogicalType::Map),
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "map".to_string(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![type_],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(None, true, 6),
                    Nested::primitive(None, false, 6),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::structure(None, true, 6),
                    Nested::primitive(None, false, 6),
                ],
            ]
        );
    }
}
