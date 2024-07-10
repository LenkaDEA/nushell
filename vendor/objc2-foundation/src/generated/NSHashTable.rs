//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;

use crate::*;

#[cfg(feature = "NSPointerFunctions")]
pub static NSHashTableStrongMemory: NSPointerFunctionsOptions =
    NSPointerFunctionsOptions(NSPointerFunctionsOptions::NSPointerFunctionsStrongMemory.0);

#[cfg(feature = "NSPointerFunctions")]
pub static NSHashTableZeroingWeakMemory: NSPointerFunctionsOptions =
    NSPointerFunctionsOptions(NSPointerFunctionsOptions::NSPointerFunctionsZeroingWeakMemory.0);

#[cfg(feature = "NSPointerFunctions")]
pub static NSHashTableCopyIn: NSPointerFunctionsOptions =
    NSPointerFunctionsOptions(NSPointerFunctionsOptions::NSPointerFunctionsCopyIn.0);

#[cfg(feature = "NSPointerFunctions")]
pub static NSHashTableObjectPointerPersonality: NSPointerFunctionsOptions =
    NSPointerFunctionsOptions(
        NSPointerFunctionsOptions::NSPointerFunctionsObjectPointerPersonality.0,
    );

#[cfg(feature = "NSPointerFunctions")]
pub static NSHashTableWeakMemory: NSPointerFunctionsOptions =
    NSPointerFunctionsOptions(NSPointerFunctionsOptions::NSPointerFunctionsWeakMemory.0);

pub type NSHashTableOptions = NSUInteger;

__inner_extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSHashTable<ObjectType: ?Sized = AnyObject> {
        __superclass: NSObject,
        _inner0: PhantomData<*mut ObjectType>,
        notunwindsafe: PhantomData<&'static mut ()>,
    }

    unsafe impl<ObjectType: ?Sized + Message> ClassType for NSHashTable<ObjectType> {
        type Super = NSObject;
        type Mutability = InteriorMutable;

        fn as_super(&self) -> &Self::Super {
            &self.__superclass
        }

        fn as_super_mut(&mut self) -> &mut Self::Super {
            &mut self.__superclass
        }
    }
);

#[cfg(feature = "NSObject")]
unsafe impl<ObjectType: ?Sized + NSCoding> NSCoding for NSHashTable<ObjectType> {}

#[cfg(feature = "NSObject")]
unsafe impl<ObjectType: ?Sized + IsIdCloneable> NSCopying for NSHashTable<ObjectType> {}

#[cfg(feature = "NSEnumerator")]
unsafe impl<ObjectType: ?Sized> NSFastEnumeration for NSHashTable<ObjectType> {}

unsafe impl<ObjectType: ?Sized> NSObjectProtocol for NSHashTable<ObjectType> {}

#[cfg(feature = "NSObject")]
unsafe impl<ObjectType: ?Sized + NSSecureCoding> NSSecureCoding for NSHashTable<ObjectType> {}

extern_methods!(
    unsafe impl<ObjectType: Message> NSHashTable<ObjectType> {
        #[cfg(feature = "NSPointerFunctions")]
        #[method_id(@__retain_semantics Init initWithOptions:capacity:)]
        pub unsafe fn initWithOptions_capacity(
            this: Allocated<Self>,
            options: NSPointerFunctionsOptions,
            initial_capacity: NSUInteger,
        ) -> Id<Self>;

        #[cfg(feature = "NSPointerFunctions")]
        #[method_id(@__retain_semantics Init initWithPointerFunctions:capacity:)]
        pub unsafe fn initWithPointerFunctions_capacity(
            this: Allocated<Self>,
            functions: &NSPointerFunctions,
            initial_capacity: NSUInteger,
        ) -> Id<Self>;

        #[cfg(feature = "NSPointerFunctions")]
        #[method_id(@__retain_semantics Other hashTableWithOptions:)]
        pub unsafe fn hashTableWithOptions(
            options: NSPointerFunctionsOptions,
        ) -> Id<NSHashTable<ObjectType>>;

        #[deprecated = "GC no longer supported"]
        #[method_id(@__retain_semantics Other hashTableWithWeakObjects)]
        pub unsafe fn hashTableWithWeakObjects() -> Id<AnyObject>;

        #[method_id(@__retain_semantics Other weakObjectsHashTable)]
        pub unsafe fn weakObjectsHashTable() -> Id<NSHashTable<ObjectType>>;

        #[cfg(feature = "NSPointerFunctions")]
        #[method_id(@__retain_semantics Other pointerFunctions)]
        pub unsafe fn pointerFunctions(&self) -> Id<NSPointerFunctions>;

        #[method(count)]
        pub unsafe fn count(&self) -> NSUInteger;

        #[method_id(@__retain_semantics Other member:)]
        pub unsafe fn member(&self, object: Option<&ObjectType>) -> Option<Id<ObjectType>>;

        #[cfg(feature = "NSEnumerator")]
        #[method_id(@__retain_semantics Other objectEnumerator)]
        pub unsafe fn objectEnumerator(&self) -> Id<NSEnumerator<ObjectType>>;

        #[method(addObject:)]
        pub unsafe fn addObject(&self, object: Option<&ObjectType>);

        #[method(removeObject:)]
        pub unsafe fn removeObject(&self, object: Option<&ObjectType>);

        #[method(removeAllObjects)]
        pub unsafe fn removeAllObjects(&self);

        #[cfg(feature = "NSArray")]
        #[method_id(@__retain_semantics Other allObjects)]
        pub unsafe fn allObjects(&self) -> Id<NSArray<ObjectType>>;

        #[method_id(@__retain_semantics Other anyObject)]
        pub unsafe fn anyObject(&self) -> Option<Id<ObjectType>>;

        #[method(containsObject:)]
        pub unsafe fn containsObject(&self, an_object: Option<&ObjectType>) -> bool;

        #[method(intersectsHashTable:)]
        pub unsafe fn intersectsHashTable(&self, other: &NSHashTable<ObjectType>) -> bool;

        #[method(isEqualToHashTable:)]
        pub unsafe fn isEqualToHashTable(&self, other: &NSHashTable<ObjectType>) -> bool;

        #[method(isSubsetOfHashTable:)]
        pub unsafe fn isSubsetOfHashTable(&self, other: &NSHashTable<ObjectType>) -> bool;

        #[method(intersectHashTable:)]
        pub unsafe fn intersectHashTable(&self, other: &NSHashTable<ObjectType>);

        #[method(unionHashTable:)]
        pub unsafe fn unionHashTable(&self, other: &NSHashTable<ObjectType>);

        #[method(minusHashTable:)]
        pub unsafe fn minusHashTable(&self, other: &NSHashTable<ObjectType>);

        #[cfg(feature = "NSSet")]
        #[method_id(@__retain_semantics Other setRepresentation)]
        pub unsafe fn setRepresentation(&self) -> Id<NSSet<ObjectType>>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl<ObjectType: Message> NSHashTable<ObjectType> {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NSHashEnumerator {
    _pi: NSUInteger,
    _si: NSUInteger,
    _bs: *mut c_void,
}

unsafe impl Encode for NSHashEnumerator {
    const ENCODING: Encoding = Encoding::Struct(
        "?",
        &[
            <NSUInteger>::ENCODING,
            <NSUInteger>::ENCODING,
            <*mut c_void>::ENCODING,
        ],
    );
}

unsafe impl RefEncode for NSHashEnumerator {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern "C" {
    pub fn NSFreeHashTable(table: &NSHashTable);
}

extern "C" {
    pub fn NSResetHashTable(table: &NSHashTable);
}

extern "C" {
    pub fn NSCompareHashTables(table1: &NSHashTable, table2: &NSHashTable) -> Bool;
}

extern "C" {
    #[cfg(feature = "NSZone")]
    pub fn NSCopyHashTableWithZone(table: &NSHashTable, zone: *mut NSZone) -> NonNull<NSHashTable>;
}

extern "C" {
    pub fn NSHashGet(table: &NSHashTable, pointer: *mut c_void) -> NonNull<c_void>;
}

extern "C" {
    pub fn NSHashInsert(table: &NSHashTable, pointer: *mut c_void);
}

extern "C" {
    pub fn NSHashInsertKnownAbsent(table: &NSHashTable, pointer: *mut c_void);
}

extern "C" {
    pub fn NSHashInsertIfAbsent(table: &NSHashTable, pointer: *mut c_void) -> *mut c_void;
}

extern "C" {
    pub fn NSHashRemove(table: &NSHashTable, pointer: *mut c_void);
}

extern "C" {
    pub fn NSEnumerateHashTable(table: &NSHashTable) -> NSHashEnumerator;
}

extern "C" {
    pub fn NSNextHashEnumeratorItem(enumerator: NonNull<NSHashEnumerator>) -> *mut c_void;
}

extern "C" {
    pub fn NSEndHashTableEnumeration(enumerator: NonNull<NSHashEnumerator>);
}

extern "C" {
    pub fn NSCountHashTable(table: &NSHashTable) -> NSUInteger;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub fn NSStringFromHashTable(table: &NSHashTable) -> NonNull<NSString>;
}

extern "C" {
    #[cfg(feature = "NSArray")]
    pub fn NSAllHashTableObjects(table: &NSHashTable) -> NonNull<NSArray>;
}

#[cfg(feature = "NSString")]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NSHashTableCallBacks {
    pub hash: Option<unsafe extern "C" fn(NonNull<NSHashTable>, NonNull<c_void>) -> NSUInteger>,
    pub isEqual: Option<
        unsafe extern "C" fn(NonNull<NSHashTable>, NonNull<c_void>, NonNull<c_void>) -> Bool,
    >,
    pub retain: Option<unsafe extern "C" fn(NonNull<NSHashTable>, NonNull<c_void>)>,
    pub release: Option<unsafe extern "C" fn(NonNull<NSHashTable>, NonNull<c_void>)>,
    pub describe:
        Option<unsafe extern "C" fn(NonNull<NSHashTable>, NonNull<c_void>) -> *mut NSString>,
}

#[cfg(feature = "NSString")]
unsafe impl Encode for NSHashTableCallBacks {
    const ENCODING: Encoding = Encoding::Struct("?", &[<Option<unsafe extern "C" fn(NonNull<NSHashTable>,NonNull<c_void>,) -> NSUInteger>>::ENCODING,<Option<unsafe extern "C" fn(NonNull<NSHashTable>,NonNull<c_void>,NonNull<c_void>,) -> Bool>>::ENCODING,<Option<unsafe extern "C" fn(NonNull<NSHashTable>,NonNull<c_void>,)>>::ENCODING,<Option<unsafe extern "C" fn(NonNull<NSHashTable>,NonNull<c_void>,)>>::ENCODING,<Option<unsafe extern "C" fn(NonNull<NSHashTable>,NonNull<c_void>,) -> *mut NSString>>::ENCODING,]);
}

#[cfg(feature = "NSString")]
unsafe impl RefEncode for NSHashTableCallBacks {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern "C" {
    #[cfg(all(feature = "NSString", feature = "NSZone"))]
    pub fn NSCreateHashTableWithZone(
        call_backs: NSHashTableCallBacks,
        capacity: NSUInteger,
        zone: *mut NSZone,
    ) -> NonNull<NSHashTable>;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub fn NSCreateHashTable(
        call_backs: NSHashTableCallBacks,
        capacity: NSUInteger,
    ) -> NonNull<NSHashTable>;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSIntegerHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSNonOwnedPointerHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSNonRetainedObjectHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSObjectHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSOwnedObjectIdentityHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSOwnedPointerHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSPointerToStructHashCallBacks: NSHashTableCallBacks;
}

extern "C" {
    #[cfg(feature = "NSString")]
    pub static NSIntHashCallBacks: NSHashTableCallBacks;
}
