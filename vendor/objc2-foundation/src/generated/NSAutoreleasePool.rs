//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;

use crate::*;

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSAutoreleasePool;

    unsafe impl ClassType for NSAutoreleasePool {
        type Super = NSObject;
        type Mutability = InteriorMutable;
    }
);

unsafe impl NSObjectProtocol for NSAutoreleasePool {}

extern_methods!(
    unsafe impl NSAutoreleasePool {
        #[method(addObject:)]
        pub unsafe fn addObject_class(an_object: &AnyObject);

        #[method(addObject:)]
        pub unsafe fn addObject(&self, an_object: &AnyObject);

        #[method(drain)]
        pub unsafe fn drain(&self);
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSAutoreleasePool {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);
