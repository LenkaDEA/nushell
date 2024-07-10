//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;
#[cfg(feature = "objc2-core-data")]
use objc2_core_data::*;
use objc2_foundation::*;

use crate::*;

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSPredicateEditorRowTemplate;

    unsafe impl ClassType for NSPredicateEditorRowTemplate {
        type Super = NSObject;
        type Mutability = InteriorMutable;
    }
);

unsafe impl NSCoding for NSPredicateEditorRowTemplate {}

unsafe impl NSCopying for NSPredicateEditorRowTemplate {}

unsafe impl NSObjectProtocol for NSPredicateEditorRowTemplate {}

extern_methods!(
    unsafe impl NSPredicateEditorRowTemplate {
        #[method(matchForPredicate:)]
        pub unsafe fn matchForPredicate(&self, predicate: &NSPredicate) -> c_double;

        #[cfg(all(feature = "NSResponder", feature = "NSView"))]
        #[method_id(@__retain_semantics Other templateViews)]
        pub unsafe fn templateViews(&self, mtm: MainThreadMarker) -> Id<NSArray<NSView>>;

        #[method(setPredicate:)]
        pub unsafe fn setPredicate(&self, predicate: &NSPredicate);

        #[method_id(@__retain_semantics Other predicateWithSubpredicates:)]
        pub unsafe fn predicateWithSubpredicates(
            &self,
            subpredicates: Option<&NSArray<NSPredicate>>,
        ) -> Id<NSPredicate>;

        #[method_id(@__retain_semantics Other displayableSubpredicatesOfPredicate:)]
        pub unsafe fn displayableSubpredicatesOfPredicate(
            &self,
            predicate: &NSPredicate,
        ) -> Option<Id<NSArray<NSPredicate>>>;

        #[method_id(@__retain_semantics Init initWithLeftExpressions:rightExpressions:modifier:operators:options:)]
        pub unsafe fn initWithLeftExpressions_rightExpressions_modifier_operators_options(
            this: Allocated<Self>,
            left_expressions: &NSArray<NSExpression>,
            right_expressions: &NSArray<NSExpression>,
            modifier: NSComparisonPredicateModifier,
            operators: &NSArray<NSNumber>,
            options: NSUInteger,
        ) -> Id<Self>;

        #[cfg(feature = "objc2-core-data")]
        #[method_id(@__retain_semantics Init initWithLeftExpressions:rightExpressionAttributeType:modifier:operators:options:)]
        pub unsafe fn initWithLeftExpressions_rightExpressionAttributeType_modifier_operators_options(
            this: Allocated<Self>,
            left_expressions: &NSArray<NSExpression>,
            attribute_type: NSAttributeType,
            modifier: NSComparisonPredicateModifier,
            operators: &NSArray<NSNumber>,
            options: NSUInteger,
        ) -> Id<Self>;

        #[method_id(@__retain_semantics Init initWithCompoundTypes:)]
        pub unsafe fn initWithCompoundTypes(
            this: Allocated<Self>,
            compound_types: &NSArray<NSNumber>,
        ) -> Id<Self>;

        #[method_id(@__retain_semantics Other leftExpressions)]
        pub unsafe fn leftExpressions(&self) -> Option<Id<NSArray<NSExpression>>>;

        #[method_id(@__retain_semantics Other rightExpressions)]
        pub unsafe fn rightExpressions(&self) -> Option<Id<NSArray<NSExpression>>>;

        #[cfg(feature = "objc2-core-data")]
        #[method(rightExpressionAttributeType)]
        pub unsafe fn rightExpressionAttributeType(&self) -> NSAttributeType;

        #[method(modifier)]
        pub unsafe fn modifier(&self) -> NSComparisonPredicateModifier;

        #[method_id(@__retain_semantics Other operators)]
        pub unsafe fn operators(&self) -> Option<Id<NSArray<NSNumber>>>;

        #[method(options)]
        pub unsafe fn options(&self) -> NSUInteger;

        #[method_id(@__retain_semantics Other compoundTypes)]
        pub unsafe fn compoundTypes(&self) -> Option<Id<NSArray<NSNumber>>>;

        #[cfg(feature = "objc2-core-data")]
        #[method_id(@__retain_semantics Other templatesWithAttributeKeyPaths:inEntityDescription:)]
        pub unsafe fn templatesWithAttributeKeyPaths_inEntityDescription(
            key_paths: &NSArray<NSString>,
            entity_description: &NSEntityDescription,
        ) -> Id<NSArray<NSPredicateEditorRowTemplate>>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSPredicateEditorRowTemplate {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new() -> Id<Self>;
    }
);
