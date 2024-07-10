//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;
use objc2_foundation::*;

use crate::*;

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSPropertyDescription;

    unsafe impl ClassType for NSPropertyDescription {
        type Super = NSObject;
        type Mutability = InteriorMutable;
    }
);

unsafe impl NSCoding for NSPropertyDescription {}

unsafe impl NSCopying for NSPropertyDescription {}

unsafe impl NSObjectProtocol for NSPropertyDescription {}

extern_methods!(
    unsafe impl NSPropertyDescription {
        #[cfg(feature = "NSEntityDescription")]
        #[method_id(@__retain_semantics Other entity)]
        pub unsafe fn entity(&self) -> Id<NSEntityDescription>;

        #[method_id(@__retain_semantics Other name)]
        pub unsafe fn name(&self) -> Id<NSString>;

        #[method(setName:)]
        pub unsafe fn setName(&self, name: &NSString);

        #[method(isOptional)]
        pub unsafe fn isOptional(&self) -> bool;

        #[method(setOptional:)]
        pub unsafe fn setOptional(&self, optional: bool);

        #[method(isTransient)]
        pub unsafe fn isTransient(&self) -> bool;

        #[method(setTransient:)]
        pub unsafe fn setTransient(&self, transient: bool);

        #[method_id(@__retain_semantics Other validationPredicates)]
        pub unsafe fn validationPredicates(&self) -> Id<NSArray<NSPredicate>>;

        #[method_id(@__retain_semantics Other validationWarnings)]
        pub unsafe fn validationWarnings(&self) -> Id<NSArray>;

        #[method(setValidationPredicates:withValidationWarnings:)]
        pub unsafe fn setValidationPredicates_withValidationWarnings(
            &self,
            validation_predicates: Option<&NSArray<NSPredicate>>,
            validation_warnings: Option<&NSArray<NSString>>,
        );

        #[method_id(@__retain_semantics Other userInfo)]
        pub unsafe fn userInfo(&self) -> Option<Id<NSDictionary>>;

        #[method(setUserInfo:)]
        pub unsafe fn setUserInfo(&self, user_info: Option<&NSDictionary>);

        #[deprecated = "Use NSEntityDescription.indexes instead"]
        #[method(isIndexed)]
        pub unsafe fn isIndexed(&self) -> bool;

        #[deprecated = "Use NSEntityDescription.indexes instead"]
        #[method(setIndexed:)]
        pub unsafe fn setIndexed(&self, indexed: bool);

        #[method_id(@__retain_semantics Other versionHash)]
        pub unsafe fn versionHash(&self) -> Id<NSData>;

        #[method_id(@__retain_semantics Other versionHashModifier)]
        pub unsafe fn versionHashModifier(&self) -> Option<Id<NSString>>;

        #[method(setVersionHashModifier:)]
        pub unsafe fn setVersionHashModifier(&self, version_hash_modifier: Option<&NSString>);

        #[method(isIndexedBySpotlight)]
        pub unsafe fn isIndexedBySpotlight(&self) -> bool;

        #[method(setIndexedBySpotlight:)]
        pub unsafe fn setIndexedBySpotlight(&self, indexed_by_spotlight: bool);

        #[deprecated = "Spotlight integration is deprecated. Use CoreSpotlight integration instead."]
        #[method(isStoredInExternalRecord)]
        pub unsafe fn isStoredInExternalRecord(&self) -> bool;

        #[deprecated = "Spotlight integration is deprecated. Use CoreSpotlight integration instead."]
        #[method(setStoredInExternalRecord:)]
        pub unsafe fn setStoredInExternalRecord(&self, stored_in_external_record: bool);

        #[method_id(@__retain_semantics Other renamingIdentifier)]
        pub unsafe fn renamingIdentifier(&self) -> Option<Id<NSString>>;

        #[method(setRenamingIdentifier:)]
        pub unsafe fn setRenamingIdentifier(&self, renaming_identifier: Option<&NSString>);
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSPropertyDescription {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);
