//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;
use objc2_foundation::*;

use crate::*;

// NS_ENUM
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSPathStyle(pub NSInteger);
impl NSPathStyle {
    #[doc(alias = "NSPathStyleStandard")]
    pub const Standard: Self = Self(0);
    #[doc(alias = "NSPathStylePopUp")]
    pub const PopUp: Self = Self(2);
    #[deprecated]
    #[doc(alias = "NSPathStyleNavigationBar")]
    pub const NavigationBar: Self = Self(1);
}

unsafe impl Encode for NSPathStyle {
    const ENCODING: Encoding = NSInteger::ENCODING;
}

unsafe impl RefEncode for NSPathStyle {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
    pub struct NSPathCell;

    #[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
    unsafe impl ClassType for NSPathCell {
        #[inherits(NSCell, NSObject)]
        type Super = NSActionCell;
        type Mutability = MainThreadOnly;
    }
);

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSActionCell",
    feature = "NSCell"
))]
unsafe impl NSAccessibility for NSPathCell {}

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSActionCell",
    feature = "NSCell"
))]
unsafe impl NSAccessibilityElementProtocol for NSPathCell {}

#[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
unsafe impl NSCoding for NSPathCell {}

#[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
unsafe impl NSCopying for NSPathCell {}

#[cfg(all(feature = "NSActionCell", feature = "NSCell", feature = "NSMenu"))]
unsafe impl NSMenuItemValidation for NSPathCell {}

#[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
unsafe impl NSObjectProtocol for NSPathCell {}

#[cfg(all(feature = "NSActionCell", feature = "NSCell", feature = "NSSavePanel"))]
unsafe impl NSOpenSavePanelDelegate for NSPathCell {}

#[cfg(all(
    feature = "NSActionCell",
    feature = "NSCell",
    feature = "NSUserInterfaceItemIdentification"
))]
unsafe impl NSUserInterfaceItemIdentification for NSPathCell {}

extern_methods!(
    #[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
    unsafe impl NSPathCell {
        #[method(pathStyle)]
        pub unsafe fn pathStyle(&self) -> NSPathStyle;

        #[method(setPathStyle:)]
        pub unsafe fn setPathStyle(&self, path_style: NSPathStyle);

        #[method_id(@__retain_semantics Other URL)]
        pub unsafe fn URL(&self) -> Option<Id<NSURL>>;

        #[method(setURL:)]
        pub unsafe fn setURL(&self, url: Option<&NSURL>);

        #[method(setObjectValue:)]
        pub unsafe fn setObjectValue(&self, obj: Option<&ProtocolObject<dyn NSCopying>>);

        #[method_id(@__retain_semantics Other allowedTypes)]
        pub unsafe fn allowedTypes(&self) -> Option<Id<NSArray<NSString>>>;

        #[method(setAllowedTypes:)]
        pub unsafe fn setAllowedTypes(&self, allowed_types: Option<&NSArray<NSString>>);

        #[method_id(@__retain_semantics Other delegate)]
        pub unsafe fn delegate(&self) -> Option<Id<ProtocolObject<dyn NSPathCellDelegate>>>;

        #[method(setDelegate:)]
        pub unsafe fn setDelegate(&self, delegate: Option<&ProtocolObject<dyn NSPathCellDelegate>>);

        #[method(pathComponentCellClass)]
        pub unsafe fn pathComponentCellClass(mtm: MainThreadMarker) -> &'static AnyClass;

        #[cfg(all(feature = "NSPathComponentCell", feature = "NSTextFieldCell"))]
        #[method_id(@__retain_semantics Other pathComponentCells)]
        pub unsafe fn pathComponentCells(&self) -> Id<NSArray<NSPathComponentCell>>;

        #[cfg(all(feature = "NSPathComponentCell", feature = "NSTextFieldCell"))]
        #[method(setPathComponentCells:)]
        pub unsafe fn setPathComponentCells(
            &self,
            path_component_cells: &NSArray<NSPathComponentCell>,
        );

        #[cfg(all(
            feature = "NSPathComponentCell",
            feature = "NSResponder",
            feature = "NSTextFieldCell",
            feature = "NSView"
        ))]
        #[method(rectOfPathComponentCell:withFrame:inView:)]
        pub unsafe fn rectOfPathComponentCell_withFrame_inView(
            &self,
            cell: &NSPathComponentCell,
            frame: NSRect,
            view: &NSView,
        ) -> NSRect;

        #[cfg(all(
            feature = "NSPathComponentCell",
            feature = "NSResponder",
            feature = "NSTextFieldCell",
            feature = "NSView"
        ))]
        #[method_id(@__retain_semantics Other pathComponentCellAtPoint:withFrame:inView:)]
        pub unsafe fn pathComponentCellAtPoint_withFrame_inView(
            &self,
            point: NSPoint,
            frame: NSRect,
            view: &NSView,
        ) -> Option<Id<NSPathComponentCell>>;

        #[cfg(all(feature = "NSPathComponentCell", feature = "NSTextFieldCell"))]
        #[method_id(@__retain_semantics Other clickedPathComponentCell)]
        pub unsafe fn clickedPathComponentCell(&self) -> Option<Id<NSPathComponentCell>>;

        #[cfg(all(feature = "NSEvent", feature = "NSResponder", feature = "NSView"))]
        #[method(mouseEntered:withFrame:inView:)]
        pub unsafe fn mouseEntered_withFrame_inView(
            &self,
            event: &NSEvent,
            frame: NSRect,
            view: &NSView,
        );

        #[cfg(all(feature = "NSEvent", feature = "NSResponder", feature = "NSView"))]
        #[method(mouseExited:withFrame:inView:)]
        pub unsafe fn mouseExited_withFrame_inView(
            &self,
            event: &NSEvent,
            frame: NSRect,
            view: &NSView,
        );

        #[method(doubleAction)]
        pub unsafe fn doubleAction(&self) -> Option<Sel>;

        #[method(setDoubleAction:)]
        pub unsafe fn setDoubleAction(&self, double_action: Option<Sel>);

        #[cfg(feature = "NSColor")]
        #[method_id(@__retain_semantics Other backgroundColor)]
        pub unsafe fn backgroundColor(&self) -> Option<Id<NSColor>>;

        #[cfg(feature = "NSColor")]
        #[method(setBackgroundColor:)]
        pub unsafe fn setBackgroundColor(&self, background_color: Option<&NSColor>);

        #[method_id(@__retain_semantics Other placeholderString)]
        pub unsafe fn placeholderString(&self) -> Option<Id<NSString>>;

        #[method(setPlaceholderString:)]
        pub unsafe fn setPlaceholderString(&self, placeholder_string: Option<&NSString>);

        #[method_id(@__retain_semantics Other placeholderAttributedString)]
        pub unsafe fn placeholderAttributedString(&self) -> Option<Id<NSAttributedString>>;

        #[method(setPlaceholderAttributedString:)]
        pub unsafe fn setPlaceholderAttributedString(
            &self,
            placeholder_attributed_string: Option<&NSAttributedString>,
        );
    }
);

extern_methods!(
    /// Methods declared on superclass `NSCell`
    #[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
    unsafe impl NSPathCell {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics Init initTextCell:)]
        pub unsafe fn initTextCell(this: Allocated<Self>, string: &NSString) -> Id<Self>;

        #[cfg(feature = "NSImage")]
        #[method_id(@__retain_semantics Init initImageCell:)]
        pub unsafe fn initImageCell(this: Allocated<Self>, image: Option<&NSImage>) -> Id<Self>;

        #[method_id(@__retain_semantics Init initWithCoder:)]
        pub unsafe fn initWithCoder(this: Allocated<Self>, coder: &NSCoder) -> Id<Self>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    #[cfg(all(feature = "NSActionCell", feature = "NSCell"))]
    unsafe impl NSPathCell {
        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new(mtm: MainThreadMarker) -> Id<Self>;
    }
);

extern_protocol!(
    pub unsafe trait NSPathCellDelegate: NSObjectProtocol + IsMainThreadOnly {
        #[cfg(all(
            feature = "NSActionCell",
            feature = "NSCell",
            feature = "NSOpenPanel",
            feature = "NSPanel",
            feature = "NSResponder",
            feature = "NSSavePanel",
            feature = "NSWindow"
        ))]
        #[optional]
        #[method(pathCell:willDisplayOpenPanel:)]
        unsafe fn pathCell_willDisplayOpenPanel(
            &self,
            path_cell: &NSPathCell,
            open_panel: &NSOpenPanel,
        );

        #[cfg(all(feature = "NSActionCell", feature = "NSCell", feature = "NSMenu"))]
        #[optional]
        #[method(pathCell:willPopUpMenu:)]
        unsafe fn pathCell_willPopUpMenu(&self, path_cell: &NSPathCell, menu: &NSMenu);
    }

    unsafe impl ProtocolType for dyn NSPathCellDelegate {}
);
