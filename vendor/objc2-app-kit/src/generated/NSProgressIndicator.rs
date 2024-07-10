//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
use objc2::__framework_prelude::*;
use objc2_foundation::*;

use crate::*;

// NS_ENUM
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSProgressIndicatorStyle(pub NSUInteger);
impl NSProgressIndicatorStyle {
    #[doc(alias = "NSProgressIndicatorStyleBar")]
    pub const Bar: Self = Self(0);
    #[doc(alias = "NSProgressIndicatorStyleSpinning")]
    pub const Spinning: Self = Self(1);
}

unsafe impl Encode for NSProgressIndicatorStyle {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSProgressIndicatorStyle {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    pub struct NSProgressIndicator;

    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl ClassType for NSProgressIndicator {
        #[inherits(NSResponder, NSObject)]
        type Super = NSView;
        type Mutability = MainThreadOnly;
    }
);

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSResponder",
    feature = "NSView"
))]
unsafe impl NSAccessibility for NSProgressIndicator {}

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSResponder",
    feature = "NSView"
))]
unsafe impl NSAccessibilityElementProtocol for NSProgressIndicator {}

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSResponder",
    feature = "NSView"
))]
unsafe impl NSAccessibilityGroup for NSProgressIndicator {}

#[cfg(all(
    feature = "NSAccessibilityProtocols",
    feature = "NSResponder",
    feature = "NSView"
))]
unsafe impl NSAccessibilityProgressIndicator for NSProgressIndicator {}

#[cfg(all(feature = "NSAnimation", feature = "NSResponder", feature = "NSView"))]
unsafe impl NSAnimatablePropertyContainer for NSProgressIndicator {}

#[cfg(all(feature = "NSAppearance", feature = "NSResponder", feature = "NSView"))]
unsafe impl NSAppearanceCustomization for NSProgressIndicator {}

#[cfg(all(feature = "NSResponder", feature = "NSView"))]
unsafe impl NSCoding for NSProgressIndicator {}

#[cfg(all(feature = "NSDragging", feature = "NSResponder", feature = "NSView"))]
unsafe impl NSDraggingDestination for NSProgressIndicator {}

#[cfg(all(feature = "NSResponder", feature = "NSView"))]
unsafe impl NSObjectProtocol for NSProgressIndicator {}

#[cfg(all(
    feature = "NSResponder",
    feature = "NSUserInterfaceItemIdentification",
    feature = "NSView"
))]
unsafe impl NSUserInterfaceItemIdentification for NSProgressIndicator {}

extern_methods!(
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl NSProgressIndicator {
        #[method(isIndeterminate)]
        pub unsafe fn isIndeterminate(&self) -> bool;

        #[method(setIndeterminate:)]
        pub unsafe fn setIndeterminate(&self, indeterminate: bool);

        #[cfg(feature = "NSCell")]
        #[method(controlSize)]
        pub unsafe fn controlSize(&self) -> NSControlSize;

        #[cfg(feature = "NSCell")]
        #[method(setControlSize:)]
        pub unsafe fn setControlSize(&self, control_size: NSControlSize);

        #[method(doubleValue)]
        pub unsafe fn doubleValue(&self) -> c_double;

        #[method(setDoubleValue:)]
        pub unsafe fn setDoubleValue(&self, double_value: c_double);

        #[method(incrementBy:)]
        pub unsafe fn incrementBy(&self, delta: c_double);

        #[method(minValue)]
        pub unsafe fn minValue(&self) -> c_double;

        #[method(setMinValue:)]
        pub unsafe fn setMinValue(&self, min_value: c_double);

        #[method(maxValue)]
        pub unsafe fn maxValue(&self) -> c_double;

        #[method(setMaxValue:)]
        pub unsafe fn setMaxValue(&self, max_value: c_double);

        #[method_id(@__retain_semantics Other observedProgress)]
        pub unsafe fn observedProgress(&self) -> Option<Id<NSProgress>>;

        #[method(setObservedProgress:)]
        pub unsafe fn setObservedProgress(&self, observed_progress: Option<&NSProgress>);

        #[method(usesThreadedAnimation)]
        pub unsafe fn usesThreadedAnimation(&self) -> bool;

        #[method(setUsesThreadedAnimation:)]
        pub unsafe fn setUsesThreadedAnimation(&self, uses_threaded_animation: bool);

        #[method(startAnimation:)]
        pub unsafe fn startAnimation(&self, sender: Option<&AnyObject>);

        #[method(stopAnimation:)]
        pub unsafe fn stopAnimation(&self, sender: Option<&AnyObject>);

        #[method(style)]
        pub unsafe fn style(&self) -> NSProgressIndicatorStyle;

        #[method(setStyle:)]
        pub unsafe fn setStyle(&self, style: NSProgressIndicatorStyle);

        #[method(sizeToFit)]
        pub unsafe fn sizeToFit(&self);

        #[method(isDisplayedWhenStopped)]
        pub unsafe fn isDisplayedWhenStopped(&self) -> bool;

        #[method(setDisplayedWhenStopped:)]
        pub unsafe fn setDisplayedWhenStopped(&self, displayed_when_stopped: bool);
    }
);

extern_methods!(
    /// Methods declared on superclass `NSView`
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl NSProgressIndicator {
        #[method_id(@__retain_semantics Init initWithFrame:)]
        pub unsafe fn initWithFrame(this: Allocated<Self>, frame_rect: NSRect) -> Id<Self>;

        #[method_id(@__retain_semantics Init initWithCoder:)]
        pub unsafe fn initWithCoder(this: Allocated<Self>, coder: &NSCoder) -> Option<Id<Self>>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSResponder`
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl NSProgressIndicator {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl NSProgressIndicator {
        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new(mtm: MainThreadMarker) -> Id<Self>;
    }
);

// NS_ENUM
#[deprecated = "These constants do not accurately represent the geometry of NSProgressIndicator.  Use `controlSize` and `sizeToFit` instead."]
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSProgressIndicatorThickness(pub NSUInteger);
impl NSProgressIndicatorThickness {
    #[deprecated = "These constants do not accurately represent the geometry of NSProgressIndicator.  Use `controlSize` and `sizeToFit` instead."]
    pub const NSProgressIndicatorPreferredThickness: Self = Self(14);
    #[deprecated = "These constants do not accurately represent the geometry of NSProgressIndicator.  Use `controlSize` and `sizeToFit` instead."]
    pub const NSProgressIndicatorPreferredSmallThickness: Self = Self(10);
    #[deprecated = "These constants do not accurately represent the geometry of NSProgressIndicator.  Use `controlSize` and `sizeToFit` instead."]
    pub const NSProgressIndicatorPreferredLargeThickness: Self = Self(18);
    #[deprecated = "These constants do not accurately represent the geometry of NSProgressIndicator.  Use `controlSize` and `sizeToFit` instead."]
    pub const NSProgressIndicatorPreferredAquaThickness: Self = Self(12);
}

unsafe impl Encode for NSProgressIndicatorThickness {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSProgressIndicatorThickness {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

pub static NSProgressIndicatorBarStyle: NSProgressIndicatorStyle =
    NSProgressIndicatorStyle(NSProgressIndicatorStyle::Bar.0);

pub static NSProgressIndicatorSpinningStyle: NSProgressIndicatorStyle =
    NSProgressIndicatorStyle(NSProgressIndicatorStyle::Spinning.0);

extern_methods!(
    /// NSProgressIndicatorDeprecated
    #[cfg(all(feature = "NSResponder", feature = "NSView"))]
    unsafe impl NSProgressIndicator {
        #[deprecated = "The animationDelay property does nothing."]
        #[method(animationDelay)]
        pub unsafe fn animationDelay(&self) -> NSTimeInterval;

        #[deprecated = "The animationDelay property does nothing."]
        #[method(setAnimationDelay:)]
        pub unsafe fn setAnimationDelay(&self, delay: NSTimeInterval);

        #[deprecated = "Use -startAnimation and -stopAnimation instead."]
        #[method(animate:)]
        pub unsafe fn animate(&self, sender: Option<&AnyObject>);

        #[deprecated = "The bezeled property is not respected on 10.15 and later"]
        #[method(isBezeled)]
        pub unsafe fn isBezeled(&self) -> bool;

        #[deprecated = "The bezeled property is not respected on 10.15 and later"]
        #[method(setBezeled:)]
        pub unsafe fn setBezeled(&self, bezeled: bool);

        #[cfg(feature = "NSCell")]
        #[deprecated = "The controlTint property is not respected on 10.15 and later"]
        #[method(controlTint)]
        pub unsafe fn controlTint(&self) -> NSControlTint;

        #[cfg(feature = "NSCell")]
        #[deprecated = "The controlTint property is not respected on 10.15 and later"]
        #[method(setControlTint:)]
        pub unsafe fn setControlTint(&self, control_tint: NSControlTint);
    }
);
