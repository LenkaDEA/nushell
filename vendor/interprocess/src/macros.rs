#![allow(unused_macros)]

/// Dispatches to a symmetrically named submodule in the target OS module.
macro_rules! impmod {
	($($osmod:ident)::+ $(as $into:ident)?) => {
		impmod!($($osmod)::+, self $(as $into)?);
	};
	($($osmod:ident)::+, $($orig:ident $(as $into:ident)?),* $(,)?) => {
		#[cfg(unix)]
		use $crate::os::unix::$($osmod)::+::{$($orig $(as $into)?,)*};
		#[cfg(windows)]
		use $crate::os::windows::$($osmod)::+::{$($orig $(as $into)?,)*};
	};
}

/// Generates a method that projects `self.0` of type `src` to a `Pin` for type `dst`.
macro_rules! pinproj_for_unpin {
	($src:ty, $dst:ty) => {
		impl $src {
			#[inline(always)]
			fn pinproj(&mut self) -> ::std::pin::Pin<&mut $dst> {
				::std::pin::Pin::new(&mut self.0)
			}
		}
	};
}

/// Calls multiple macros, passing the same identifer or type as well as optional per-macro
/// parameters.
///
/// The identifier or type goes first, then comma-separated macro names without exclamation points.
/// To pass per-macro parameters, encase them in parentheses.
macro_rules! multimacro {
	($pre:tt $ty:ident, $($macro:ident $(($($arg:tt)+))?),+ $(,)?) => {$(
		$macro!($pre $ty $(, $($arg)+)?);
	)+};
	($pre:tt $ty:ty, $($macro:ident $(($($arg:tt)+))?),+ $(,)?) => {$(
		$macro!($pre $ty $(, $($arg)+)?);
	)+};
	($ty:ident, $($macro:ident $(($($arg:tt)+))?),+ $(,)?) => {$(
		$macro!($ty $(, $($arg)+)?);
	)+};
	($ty:ty, $($macro:ident $(($($arg:tt)+))?),+ $(,)?) => {$(
		$macro!($ty $(, $($arg)+)?);
	)+};
}

/// Generates a method that immutably borrows `self.0` of type `int` for type `ty`.
///
/// If `kind` is `&`, `self.0` is borrowed directly. If `kind` is `*`, `self.0` is treated as a
/// smart pointer (`Deref` is applied).
///
/// The method generated by this macro is used by forwarding macros.
macro_rules! forward_rbv {
	(@$slf:ident, &) => { &$slf.0 };
	(@$slf:ident, *) => { &&*$slf.0 };
	($ty:ty, $int:ty, $kind:tt) => {
		impl $ty {
			#[inline(always)]
			fn refwd(&self) -> &$int {
				forward_rbv!(@self, $kind)
			}
		}
	};
}

#[rustfmt::skip] macro_rules! builder_must_use {() => {
"builder setters take the entire structure and return it with the corresponding field modified"
};}

/// Generates public self-by-value setters for builder structures. Assumes that a field of the same
/// name is public on `Self`.
macro_rules! builder_setters {
	($(#[doc = $($doc:expr)+])+ $name:ident : $ty:ty) => {
		$(#[doc = $($doc)+])+
		#[must_use = builder_must_use!()]
		#[inline(always)]
		pub fn $name(mut self, $name: $ty) -> Self {
			self.$name = $name.into();
			self
		}

	};
	($name:ident : $ty:ty) => {
		builder_setters!(
			#[doc = concat!(
				"Sets the [`",
				stringify!($name),
				"`](#structfield.", stringify!($name),
				") parameter to the specified value."
			)]
			$name : $ty
		);
	};
	($($(#[doc = $($doc:expr)+])* $name:ident : $ty:ty),+ $(,)?) => {
		$(builder_setters!($(#[doc = $($doc)+])* $name: $ty);)+
	};
}

/// Creates a public sealed uninhabited type with a bunch of unnecessary trait implementations.
macro_rules! tag_enum {
	($($(#[$attr:meta])* $tag:ident),+ $(,)?) => {$(
		$( #[$attr] )*
		#[derive(Copy, Clone, Debug, PartialEq, Eq)]
		pub enum $tag {}
		impl $crate::Sealed for $tag {}
	)+};
}

/// Generates this module's macro submodules.
macro_rules! make_macro_modules {
	($($modname:ident),+ $(,)?) => {$(
		#[macro_use] mod $modname;
		#[allow(unused_imports)]
		pub(crate) use $modname::*;
	)+};
}

make_macro_modules! {
	derive_raw, derive_mut_iorw, derive_trivconv,
	forward_handle_and_fd, forward_try_clone, forward_to_self, forward_iorw, forward_fmt,
}
