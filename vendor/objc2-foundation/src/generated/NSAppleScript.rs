//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;

use crate::*;

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSAppleScriptErrorMessage: &'static NSString;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSAppleScriptErrorNumber: &'static NSString;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSAppleScriptErrorAppName: &'static NSString;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSAppleScriptErrorBriefMessage: &'static NSString;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSAppleScriptErrorRange: &'static NSString;
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSAppleScript;

    unsafe impl ClassType for NSAppleScript {
        type Super = NSObject;
        type Mutability = InteriorMutable;
    }
);

#[cfg(feature = "NSObject")]
unsafe impl NSCopying for NSAppleScript {}

unsafe impl NSObjectProtocol for NSAppleScript {}

extern_methods!(
    unsafe impl NSAppleScript {
        #[cfg(all(feature = "NSDictionary", feature = "NSString", feature = "NSURL"))]
        #[method_id(@__retain_semantics Init initWithContentsOfURL:error:)]
        pub unsafe fn initWithContentsOfURL_error(
            this: Allocated<Self>,
            url: &NSURL,
            error_info: Option<&mut Option<Id<NSDictionary<NSString, AnyObject>>>>,
        ) -> Option<Id<Self>>;

        #[cfg(feature = "NSString")]
        #[method_id(@__retain_semantics Init initWithSource:)]
        pub unsafe fn initWithSource(this: Allocated<Self>, source: &NSString) -> Option<Id<Self>>;

        #[cfg(feature = "NSString")]
        #[method_id(@__retain_semantics Other source)]
        pub unsafe fn source(&self) -> Option<Id<NSString>>;

        #[method(isCompiled)]
        pub unsafe fn isCompiled(&self) -> bool;

        #[cfg(all(feature = "NSDictionary", feature = "NSString"))]
        #[method(compileAndReturnError:)]
        pub unsafe fn compileAndReturnError(
            &self,
            error_info: Option<&mut Option<Id<NSDictionary<NSString, AnyObject>>>>,
        ) -> bool;

        #[cfg(all(
            feature = "NSAppleEventDescriptor",
            feature = "NSDictionary",
            feature = "NSString"
        ))]
        #[method_id(@__retain_semantics Other executeAndReturnError:)]
        pub unsafe fn executeAndReturnError(
            &self,
            error_info: Option<&mut Option<Id<NSDictionary<NSString, AnyObject>>>>,
        ) -> Id<NSAppleEventDescriptor>;

        #[cfg(all(
            feature = "NSAppleEventDescriptor",
            feature = "NSDictionary",
            feature = "NSString"
        ))]
        #[method_id(@__retain_semantics Other executeAppleEvent:error:)]
        pub unsafe fn executeAppleEvent_error(
            &self,
            event: &NSAppleEventDescriptor,
            error_info: Option<&mut Option<Id<NSDictionary<NSString, AnyObject>>>>,
        ) -> Id<NSAppleEventDescriptor>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSAppleScript {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);
