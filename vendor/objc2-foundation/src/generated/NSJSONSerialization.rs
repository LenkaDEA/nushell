//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;

use crate::*;

// NS_OPTIONS
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSJSONReadingOptions(pub NSUInteger);
impl NSJSONReadingOptions {
    pub const NSJSONReadingMutableContainers: Self = Self(1 << 0);
    pub const NSJSONReadingMutableLeaves: Self = Self(1 << 1);
    pub const NSJSONReadingFragmentsAllowed: Self = Self(1 << 2);
    pub const NSJSONReadingJSON5Allowed: Self = Self(1 << 3);
    pub const NSJSONReadingTopLevelDictionaryAssumed: Self = Self(1 << 4);
    #[deprecated]
    pub const NSJSONReadingAllowFragments: Self =
        Self(NSJSONReadingOptions::NSJSONReadingFragmentsAllowed.0);
}

unsafe impl Encode for NSJSONReadingOptions {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSJSONReadingOptions {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

// NS_OPTIONS
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSJSONWritingOptions(pub NSUInteger);
impl NSJSONWritingOptions {
    pub const NSJSONWritingPrettyPrinted: Self = Self(1 << 0);
    pub const NSJSONWritingSortedKeys: Self = Self(1 << 1);
    pub const NSJSONWritingFragmentsAllowed: Self = Self(1 << 2);
    pub const NSJSONWritingWithoutEscapingSlashes: Self = Self(1 << 3);
}

unsafe impl Encode for NSJSONWritingOptions {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSJSONWritingOptions {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSJSONSerialization;

    unsafe impl ClassType for NSJSONSerialization {
        type Super = NSObject;
        type Mutability = InteriorMutable;
    }
);

unsafe impl NSObjectProtocol for NSJSONSerialization {}

extern_methods!(
    unsafe impl NSJSONSerialization {
        #[method(isValidJSONObject:)]
        pub unsafe fn isValidJSONObject(obj: &AnyObject) -> bool;

        #[cfg(all(feature = "NSData", feature = "NSError"))]
        #[method_id(@__retain_semantics Other dataWithJSONObject:options:error:_)]
        pub unsafe fn dataWithJSONObject_options_error(
            obj: &AnyObject,
            opt: NSJSONWritingOptions,
        ) -> Result<Id<NSData>, Id<NSError>>;

        #[cfg(all(feature = "NSData", feature = "NSError"))]
        #[method_id(@__retain_semantics Other JSONObjectWithData:options:error:_)]
        pub unsafe fn JSONObjectWithData_options_error(
            data: &NSData,
            opt: NSJSONReadingOptions,
        ) -> Result<Id<AnyObject>, Id<NSError>>;

        #[cfg(all(feature = "NSError", feature = "NSStream"))]
        #[method_id(@__retain_semantics Other JSONObjectWithStream:options:error:_)]
        pub unsafe fn JSONObjectWithStream_options_error(
            stream: &NSInputStream,
            opt: NSJSONReadingOptions,
        ) -> Result<Id<AnyObject>, Id<NSError>>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSJSONSerialization {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);
