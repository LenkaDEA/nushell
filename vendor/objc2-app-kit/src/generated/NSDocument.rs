//! This file has been automatically generated by `objc2`'s `header-translator`.
//! DO NOT EDIT
#[cfg(feature = "block2")]
use block2::*;
use objc2::__framework_prelude::*;
use objc2_foundation::*;

use crate::*;

// NS_ENUM
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSDocumentChangeType(pub NSUInteger);
impl NSDocumentChangeType {
    pub const NSChangeDone: Self = Self(0);
    pub const NSChangeUndone: Self = Self(1);
    pub const NSChangeRedone: Self = Self(5);
    pub const NSChangeCleared: Self = Self(2);
    pub const NSChangeReadOtherContents: Self = Self(3);
    pub const NSChangeAutosaved: Self = Self(4);
    pub const NSChangeDiscardable: Self = Self(256);
}

unsafe impl Encode for NSDocumentChangeType {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSDocumentChangeType {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

// NS_ENUM
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NSSaveOperationType(pub NSUInteger);
impl NSSaveOperationType {
    pub const NSSaveOperation: Self = Self(0);
    pub const NSSaveAsOperation: Self = Self(1);
    pub const NSSaveToOperation: Self = Self(2);
    pub const NSAutosaveInPlaceOperation: Self = Self(4);
    pub const NSAutosaveElsewhereOperation: Self = Self(3);
    pub const NSAutosaveAsOperation: Self = Self(5);
    #[deprecated = "Use NSAutosaveElsewhereOperation instead"]
    pub const NSAutosaveOperation: Self = Self(3);
}

unsafe impl Encode for NSSaveOperationType {
    const ENCODING: Encoding = NSUInteger::ENCODING;
}

unsafe impl RefEncode for NSSaveOperationType {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Self::ENCODING);
}

extern_class!(
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct NSDocument;

    unsafe impl ClassType for NSDocument {
        type Super = NSObject;
        type Mutability = MainThreadOnly;
    }
);

#[cfg(feature = "NSKeyValueBinding")]
unsafe impl NSEditorRegistration for NSDocument {}

unsafe impl NSFilePresenter for NSDocument {}

#[cfg(feature = "NSMenu")]
unsafe impl NSMenuItemValidation for NSDocument {}

unsafe impl NSObjectProtocol for NSDocument {}

#[cfg(feature = "NSUserInterfaceValidation")]
unsafe impl NSUserInterfaceValidations for NSDocument {}

extern_methods!(
    unsafe impl NSDocument {
        #[method_id(@__retain_semantics Init init)]
        pub unsafe fn init(this: Allocated<Self>) -> Id<Self>;

        #[method_id(@__retain_semantics Init initWithType:error:_)]
        pub unsafe fn initWithType_error(
            this: Allocated<Self>,
            type_name: &NSString,
        ) -> Result<Id<Self>, Id<NSError>>;

        #[method(canConcurrentlyReadDocumentsOfType:)]
        pub unsafe fn canConcurrentlyReadDocumentsOfType(
            type_name: &NSString,
            mtm: MainThreadMarker,
        ) -> bool;

        #[method_id(@__retain_semantics Init initWithContentsOfURL:ofType:error:_)]
        pub unsafe fn initWithContentsOfURL_ofType_error(
            this: Allocated<Self>,
            url: &NSURL,
            type_name: &NSString,
        ) -> Result<Id<Self>, Id<NSError>>;

        #[method_id(@__retain_semantics Init initForURL:withContentsOfURL:ofType:error:_)]
        pub unsafe fn initForURL_withContentsOfURL_ofType_error(
            this: Allocated<Self>,
            url_or_nil: Option<&NSURL>,
            contents_url: &NSURL,
            type_name: &NSString,
        ) -> Result<Id<Self>, Id<NSError>>;

        #[method_id(@__retain_semantics Other fileType)]
        pub unsafe fn fileType(&self) -> Option<Id<NSString>>;

        #[method(setFileType:)]
        pub unsafe fn setFileType(&self, file_type: Option<&NSString>);

        #[method_id(@__retain_semantics Other fileURL)]
        pub unsafe fn fileURL(&self) -> Option<Id<NSURL>>;

        #[method(setFileURL:)]
        pub unsafe fn setFileURL(&self, file_url: Option<&NSURL>);

        #[method_id(@__retain_semantics Other fileModificationDate)]
        pub unsafe fn fileModificationDate(&self) -> Option<Id<NSDate>>;

        #[method(setFileModificationDate:)]
        pub unsafe fn setFileModificationDate(&self, file_modification_date: Option<&NSDate>);

        #[method(isDraft)]
        pub unsafe fn isDraft(&self) -> bool;

        #[method(setDraft:)]
        pub unsafe fn setDraft(&self, draft: bool);

        #[cfg(feature = "block2")]
        #[method(performActivityWithSynchronousWaiting:usingBlock:)]
        pub unsafe fn performActivityWithSynchronousWaiting_usingBlock(
            &self,
            wait_synchronously: bool,
            block: &Block<dyn Fn(NonNull<Block<dyn Fn()>>)>,
        );

        #[cfg(feature = "block2")]
        #[method(continueActivityUsingBlock:)]
        pub unsafe fn continueActivityUsingBlock(&self, block: &Block<dyn Fn() + '_>);

        #[cfg(feature = "block2")]
        #[method(continueAsynchronousWorkOnMainThreadUsingBlock:)]
        pub unsafe fn continueAsynchronousWorkOnMainThreadUsingBlock(
            &self,
            block: &Block<dyn Fn()>,
        );

        #[cfg(feature = "block2")]
        #[method(performSynchronousFileAccessUsingBlock:)]
        pub unsafe fn performSynchronousFileAccessUsingBlock(&self, block: &Block<dyn Fn() + '_>);

        #[cfg(feature = "block2")]
        #[method(performAsynchronousFileAccessUsingBlock:)]
        pub unsafe fn performAsynchronousFileAccessUsingBlock(
            &self,
            block: &Block<dyn Fn(NonNull<Block<dyn Fn()>>)>,
        );

        #[method(revertDocumentToSaved:)]
        pub unsafe fn revertDocumentToSaved(&self, sender: Option<&AnyObject>);

        #[method(revertToContentsOfURL:ofType:error:_)]
        pub unsafe fn revertToContentsOfURL_ofType_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
        ) -> Result<(), Id<NSError>>;

        #[method(readFromURL:ofType:error:_)]
        pub unsafe fn readFromURL_ofType_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
        ) -> Result<(), Id<NSError>>;

        #[method(readFromFileWrapper:ofType:error:_)]
        pub unsafe fn readFromFileWrapper_ofType_error(
            &self,
            file_wrapper: &NSFileWrapper,
            type_name: &NSString,
        ) -> Result<(), Id<NSError>>;

        #[method(readFromData:ofType:error:_)]
        pub unsafe fn readFromData_ofType_error(
            &self,
            data: &NSData,
            type_name: &NSString,
        ) -> Result<(), Id<NSError>>;

        #[method(isEntireFileLoaded)]
        pub unsafe fn isEntireFileLoaded(&self) -> bool;

        #[method(writeToURL:ofType:error:_)]
        pub unsafe fn writeToURL_ofType_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
        ) -> Result<(), Id<NSError>>;

        #[method_id(@__retain_semantics Other fileWrapperOfType:error:_)]
        pub unsafe fn fileWrapperOfType_error(
            &self,
            type_name: &NSString,
        ) -> Result<Id<NSFileWrapper>, Id<NSError>>;

        #[method_id(@__retain_semantics Other dataOfType:error:_)]
        pub unsafe fn dataOfType_error(
            &self,
            type_name: &NSString,
        ) -> Result<Id<NSData>, Id<NSError>>;

        #[method(unblockUserInteraction)]
        pub unsafe fn unblockUserInteraction(&self);

        #[method(autosavingIsImplicitlyCancellable)]
        pub unsafe fn autosavingIsImplicitlyCancellable(&self) -> bool;

        #[method(writeSafelyToURL:ofType:forSaveOperation:error:_)]
        pub unsafe fn writeSafelyToURL_ofType_forSaveOperation_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
        ) -> Result<(), Id<NSError>>;

        #[method(writeToURL:ofType:forSaveOperation:originalContentsURL:error:_)]
        pub unsafe fn writeToURL_ofType_forSaveOperation_originalContentsURL_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
            absolute_original_contents_url: Option<&NSURL>,
        ) -> Result<(), Id<NSError>>;

        #[method_id(@__retain_semantics Other fileAttributesToWriteToURL:ofType:forSaveOperation:originalContentsURL:error:_)]
        pub unsafe fn fileAttributesToWriteToURL_ofType_forSaveOperation_originalContentsURL_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
            absolute_original_contents_url: Option<&NSURL>,
        ) -> Result<Id<NSDictionary<NSString, AnyObject>>, Id<NSError>>;

        #[method(keepBackupFile)]
        pub unsafe fn keepBackupFile(&self) -> bool;

        #[method_id(@__retain_semantics Other backupFileURL)]
        pub unsafe fn backupFileURL(&self) -> Option<Id<NSURL>>;

        #[method(saveDocument:)]
        pub unsafe fn saveDocument(&self, sender: Option<&AnyObject>);

        #[method(saveDocumentAs:)]
        pub unsafe fn saveDocumentAs(&self, sender: Option<&AnyObject>);

        #[method(saveDocumentTo:)]
        pub unsafe fn saveDocumentTo(&self, sender: Option<&AnyObject>);

        #[method(saveDocumentWithDelegate:didSaveSelector:contextInfo:)]
        pub unsafe fn saveDocumentWithDelegate_didSaveSelector_contextInfo(
            &self,
            delegate: Option<&AnyObject>,
            did_save_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method(runModalSavePanelForSaveOperation:delegate:didSaveSelector:contextInfo:)]
        pub unsafe fn runModalSavePanelForSaveOperation_delegate_didSaveSelector_contextInfo(
            &self,
            save_operation: NSSaveOperationType,
            delegate: Option<&AnyObject>,
            did_save_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method(shouldRunSavePanelWithAccessoryView)]
        pub unsafe fn shouldRunSavePanelWithAccessoryView(&self) -> bool;

        #[cfg(all(
            feature = "NSPanel",
            feature = "NSResponder",
            feature = "NSSavePanel",
            feature = "NSWindow"
        ))]
        #[method(prepareSavePanel:)]
        pub unsafe fn prepareSavePanel(&self, save_panel: &NSSavePanel) -> bool;

        #[method(fileNameExtensionWasHiddenInLastRunSavePanel)]
        pub unsafe fn fileNameExtensionWasHiddenInLastRunSavePanel(&self) -> bool;

        #[method_id(@__retain_semantics Other fileTypeFromLastRunSavePanel)]
        pub unsafe fn fileTypeFromLastRunSavePanel(&self) -> Option<Id<NSString>>;

        #[method(saveToURL:ofType:forSaveOperation:delegate:didSaveSelector:contextInfo:)]
        pub unsafe fn saveToURL_ofType_forSaveOperation_delegate_didSaveSelector_contextInfo(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
            delegate: Option<&AnyObject>,
            did_save_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[cfg(feature = "block2")]
        #[method(saveToURL:ofType:forSaveOperation:completionHandler:)]
        pub unsafe fn saveToURL_ofType_forSaveOperation_completionHandler(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
            completion_handler: &Block<dyn Fn(*mut NSError)>,
        );

        #[method(canAsynchronouslyWriteToURL:ofType:forSaveOperation:)]
        pub unsafe fn canAsynchronouslyWriteToURL_ofType_forSaveOperation(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
        ) -> bool;

        #[method(checkAutosavingSafetyAndReturnError:_)]
        pub unsafe fn checkAutosavingSafetyAndReturnError(&self) -> Result<(), Id<NSError>>;

        #[method(scheduleAutosaving)]
        pub unsafe fn scheduleAutosaving(&self);

        #[method(hasUnautosavedChanges)]
        pub unsafe fn hasUnautosavedChanges(&self) -> bool;

        #[method(autosaveDocumentWithDelegate:didAutosaveSelector:contextInfo:)]
        pub unsafe fn autosaveDocumentWithDelegate_didAutosaveSelector_contextInfo(
            &self,
            delegate: Option<&AnyObject>,
            did_autosave_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[cfg(feature = "block2")]
        #[method(autosaveWithImplicitCancellability:completionHandler:)]
        pub unsafe fn autosaveWithImplicitCancellability_completionHandler(
            &self,
            autosaving_is_implicitly_cancellable: bool,
            completion_handler: &Block<dyn Fn(*mut NSError)>,
        );

        #[method(autosavesInPlace)]
        pub unsafe fn autosavesInPlace(mtm: MainThreadMarker) -> bool;

        #[method(preservesVersions)]
        pub unsafe fn preservesVersions(mtm: MainThreadMarker) -> bool;

        #[method(browseDocumentVersions:)]
        pub unsafe fn browseDocumentVersions(&self, sender: Option<&AnyObject>);

        #[method(isBrowsingVersions)]
        pub unsafe fn isBrowsingVersions(&self) -> bool;

        #[cfg(feature = "block2")]
        #[method(stopBrowsingVersionsWithCompletionHandler:)]
        pub unsafe fn stopBrowsingVersionsWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn()>>,
        );

        #[method(autosavesDrafts)]
        pub unsafe fn autosavesDrafts(mtm: MainThreadMarker) -> bool;

        #[method_id(@__retain_semantics Other autosavingFileType)]
        pub unsafe fn autosavingFileType(&self) -> Option<Id<NSString>>;

        #[method_id(@__retain_semantics Other autosavedContentsFileURL)]
        pub unsafe fn autosavedContentsFileURL(&self) -> Option<Id<NSURL>>;

        #[method(setAutosavedContentsFileURL:)]
        pub unsafe fn setAutosavedContentsFileURL(
            &self,
            autosaved_contents_file_url: Option<&NSURL>,
        );

        #[method(canCloseDocumentWithDelegate:shouldCloseSelector:contextInfo:)]
        pub unsafe fn canCloseDocumentWithDelegate_shouldCloseSelector_contextInfo(
            &self,
            delegate: &AnyObject,
            should_close_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method(close)]
        pub unsafe fn close(&self);

        #[method(duplicateDocument:)]
        pub unsafe fn duplicateDocument(&self, sender: Option<&AnyObject>);

        #[method(duplicateDocumentWithDelegate:didDuplicateSelector:contextInfo:)]
        pub unsafe fn duplicateDocumentWithDelegate_didDuplicateSelector_contextInfo(
            &self,
            delegate: Option<&AnyObject>,
            did_duplicate_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method_id(@__retain_semantics Other duplicateAndReturnError:_)]
        pub unsafe fn duplicateAndReturnError(&self) -> Result<Id<NSDocument>, Id<NSError>>;

        #[method(renameDocument:)]
        pub unsafe fn renameDocument(&self, sender: Option<&AnyObject>);

        #[method(moveDocumentToUbiquityContainer:)]
        pub unsafe fn moveDocumentToUbiquityContainer(&self, sender: Option<&AnyObject>);

        #[method(moveDocument:)]
        pub unsafe fn moveDocument(&self, sender: Option<&AnyObject>);

        #[cfg(feature = "block2")]
        #[method(moveDocumentWithCompletionHandler:)]
        pub unsafe fn moveDocumentWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn(Bool)>>,
        );

        #[cfg(feature = "block2")]
        #[method(moveToURL:completionHandler:)]
        pub unsafe fn moveToURL_completionHandler(
            &self,
            url: &NSURL,
            completion_handler: Option<&Block<dyn Fn(*mut NSError)>>,
        );

        #[method(lockDocument:)]
        pub unsafe fn lockDocument(&self, sender: Option<&AnyObject>);

        #[method(unlockDocument:)]
        pub unsafe fn unlockDocument(&self, sender: Option<&AnyObject>);

        #[cfg(feature = "block2")]
        #[method(lockDocumentWithCompletionHandler:)]
        pub unsafe fn lockDocumentWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn(Bool)>>,
        );

        #[cfg(feature = "block2")]
        #[method(lockWithCompletionHandler:)]
        pub unsafe fn lockWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn(*mut NSError)>>,
        );

        #[cfg(feature = "block2")]
        #[method(unlockDocumentWithCompletionHandler:)]
        pub unsafe fn unlockDocumentWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn(Bool)>>,
        );

        #[cfg(feature = "block2")]
        #[method(unlockWithCompletionHandler:)]
        pub unsafe fn unlockWithCompletionHandler(
            &self,
            completion_handler: Option<&Block<dyn Fn(*mut NSError)>>,
        );

        #[method(isLocked)]
        pub unsafe fn isLocked(&self) -> bool;

        #[method(runPageLayout:)]
        pub unsafe fn runPageLayout(&self, sender: Option<&AnyObject>);

        #[cfg(feature = "NSPrintInfo")]
        #[method(runModalPageLayoutWithPrintInfo:delegate:didRunSelector:contextInfo:)]
        pub unsafe fn runModalPageLayoutWithPrintInfo_delegate_didRunSelector_contextInfo(
            &self,
            print_info: &NSPrintInfo,
            delegate: Option<&AnyObject>,
            did_run_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[cfg(feature = "NSPageLayout")]
        #[method(preparePageLayout:)]
        pub unsafe fn preparePageLayout(&self, page_layout: &NSPageLayout) -> bool;

        #[cfg(feature = "NSPrintInfo")]
        #[method(shouldChangePrintInfo:)]
        pub unsafe fn shouldChangePrintInfo(&self, new_print_info: &NSPrintInfo) -> bool;

        #[cfg(feature = "NSPrintInfo")]
        #[method_id(@__retain_semantics Other printInfo)]
        pub unsafe fn printInfo(&self) -> Id<NSPrintInfo>;

        #[cfg(feature = "NSPrintInfo")]
        #[method(setPrintInfo:)]
        pub unsafe fn setPrintInfo(&self, print_info: &NSPrintInfo);

        #[method(printDocument:)]
        pub unsafe fn printDocument(&self, sender: Option<&AnyObject>);

        #[cfg(feature = "NSPrintInfo")]
        #[method(printDocumentWithSettings:showPrintPanel:delegate:didPrintSelector:contextInfo:)]
        pub unsafe fn printDocumentWithSettings_showPrintPanel_delegate_didPrintSelector_contextInfo(
            &self,
            print_settings: &NSDictionary<NSPrintInfoAttributeKey, AnyObject>,
            show_print_panel: bool,
            delegate: Option<&AnyObject>,
            did_print_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[cfg(all(feature = "NSPrintInfo", feature = "NSPrintOperation"))]
        #[method_id(@__retain_semantics Other printOperationWithSettings:error:_)]
        pub unsafe fn printOperationWithSettings_error(
            &self,
            print_settings: &NSDictionary<NSPrintInfoAttributeKey, AnyObject>,
        ) -> Result<Id<NSPrintOperation>, Id<NSError>>;

        #[cfg(feature = "NSPrintOperation")]
        #[method(runModalPrintOperation:delegate:didRunSelector:contextInfo:)]
        pub unsafe fn runModalPrintOperation_delegate_didRunSelector_contextInfo(
            &self,
            print_operation: &NSPrintOperation,
            delegate: Option<&AnyObject>,
            did_run_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method(saveDocumentToPDF:)]
        pub unsafe fn saveDocumentToPDF(&self, sender: Option<&AnyObject>);

        #[cfg(feature = "NSPrintOperation")]
        #[method_id(@__retain_semantics Other PDFPrintOperation)]
        pub unsafe fn PDFPrintOperation(&self) -> Id<NSPrintOperation>;

        #[method(allowsDocumentSharing)]
        pub unsafe fn allowsDocumentSharing(&self) -> bool;

        #[cfg(all(feature = "NSSharingService", feature = "block2"))]
        #[method(shareDocumentWithSharingService:completionHandler:)]
        pub unsafe fn shareDocumentWithSharingService_completionHandler(
            &self,
            sharing_service: &NSSharingService,
            completion_handler: Option<&Block<dyn Fn(Bool)>>,
        );

        #[cfg(feature = "NSSharingService")]
        #[method(prepareSharingServicePicker:)]
        pub unsafe fn prepareSharingServicePicker(
            &self,
            sharing_service_picker: &NSSharingServicePicker,
        );

        #[cfg(feature = "NSPreviewRepresentingActivityItem")]
        #[method_id(@__retain_semantics Other previewRepresentableActivityItems)]
        pub unsafe fn previewRepresentableActivityItems(
            &self,
        ) -> Option<Id<NSArray<ProtocolObject<dyn NSPreviewRepresentableActivityItem>>>>;

        #[cfg(feature = "NSPreviewRepresentingActivityItem")]
        #[method(setPreviewRepresentableActivityItems:)]
        pub unsafe fn setPreviewRepresentableActivityItems(
            &self,
            preview_representable_activity_items: Option<
                &NSArray<ProtocolObject<dyn NSPreviewRepresentableActivityItem>>,
            >,
        );

        #[method(isDocumentEdited)]
        pub unsafe fn isDocumentEdited(&self) -> bool;

        #[method(isInViewingMode)]
        pub unsafe fn isInViewingMode(&self) -> bool;

        #[method(updateChangeCount:)]
        pub unsafe fn updateChangeCount(&self, change: NSDocumentChangeType);

        #[method_id(@__retain_semantics Other changeCountTokenForSaveOperation:)]
        pub unsafe fn changeCountTokenForSaveOperation(
            &self,
            save_operation: NSSaveOperationType,
        ) -> Id<AnyObject>;

        #[method(updateChangeCountWithToken:forSaveOperation:)]
        pub unsafe fn updateChangeCountWithToken_forSaveOperation(
            &self,
            change_count_token: &AnyObject,
            save_operation: NSSaveOperationType,
        );

        #[method_id(@__retain_semantics Other undoManager)]
        pub unsafe fn undoManager(&self) -> Option<Id<NSUndoManager>>;

        #[method(setUndoManager:)]
        pub unsafe fn setUndoManager(&self, undo_manager: Option<&NSUndoManager>);

        #[method(hasUndoManager)]
        pub unsafe fn hasUndoManager(&self) -> bool;

        #[method(setHasUndoManager:)]
        pub unsafe fn setHasUndoManager(&self, has_undo_manager: bool);

        #[cfg(all(feature = "NSResponder", feature = "NSWindow"))]
        #[method(presentError:modalForWindow:delegate:didPresentSelector:contextInfo:)]
        pub unsafe fn presentError_modalForWindow_delegate_didPresentSelector_contextInfo(
            &self,
            error: &NSError,
            window: &NSWindow,
            delegate: Option<&AnyObject>,
            did_present_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method(presentError:)]
        pub unsafe fn presentError(&self, error: &NSError) -> bool;

        #[method_id(@__retain_semantics Other willPresentError:)]
        pub unsafe fn willPresentError(&self, error: &NSError) -> Id<NSError>;

        #[method(willNotPresentError:)]
        pub unsafe fn willNotPresentError(&self, error: &NSError);

        #[method(makeWindowControllers)]
        pub unsafe fn makeWindowControllers(&self);

        #[cfg(feature = "NSNib")]
        #[method_id(@__retain_semantics Other windowNibName)]
        pub unsafe fn windowNibName(&self) -> Option<Id<NSNibName>>;

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method(windowControllerWillLoadNib:)]
        pub unsafe fn windowControllerWillLoadNib(&self, window_controller: &NSWindowController);

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method(windowControllerDidLoadNib:)]
        pub unsafe fn windowControllerDidLoadNib(&self, window_controller: &NSWindowController);

        #[cfg(all(feature = "NSResponder", feature = "NSWindow"))]
        #[method(setWindow:)]
        pub unsafe fn setWindow(&self, window: Option<&NSWindow>);

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method(addWindowController:)]
        pub unsafe fn addWindowController(&self, window_controller: &NSWindowController);

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method(removeWindowController:)]
        pub unsafe fn removeWindowController(&self, window_controller: &NSWindowController);

        #[method(showWindows)]
        pub unsafe fn showWindows(&self);

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method_id(@__retain_semantics Other windowControllers)]
        pub unsafe fn windowControllers(&self) -> Id<NSArray<NSWindowController>>;

        #[cfg(all(feature = "NSResponder", feature = "NSWindowController"))]
        #[method(shouldCloseWindowController:delegate:shouldCloseSelector:contextInfo:)]
        pub unsafe fn shouldCloseWindowController_delegate_shouldCloseSelector_contextInfo(
            &self,
            window_controller: &NSWindowController,
            delegate: Option<&AnyObject>,
            should_close_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[method_id(@__retain_semantics Other displayName)]
        pub unsafe fn displayName(&self) -> Id<NSString>;

        #[method_id(@__retain_semantics Other defaultDraftName)]
        pub unsafe fn defaultDraftName(&self) -> Id<NSString>;

        #[cfg(all(feature = "NSResponder", feature = "NSWindow"))]
        #[method_id(@__retain_semantics Other windowForSheet)]
        pub unsafe fn windowForSheet(&self) -> Option<Id<NSWindow>>;

        #[method_id(@__retain_semantics Other readableTypes)]
        pub unsafe fn readableTypes(mtm: MainThreadMarker) -> Id<NSArray<NSString>>;

        #[method_id(@__retain_semantics Other writableTypes)]
        pub unsafe fn writableTypes(mtm: MainThreadMarker) -> Id<NSArray<NSString>>;

        #[method(isNativeType:)]
        pub unsafe fn isNativeType(r#type: &NSString, mtm: MainThreadMarker) -> bool;

        #[method_id(@__retain_semantics Other writableTypesForSaveOperation:)]
        pub unsafe fn writableTypesForSaveOperation(
            &self,
            save_operation: NSSaveOperationType,
        ) -> Id<NSArray<NSString>>;

        #[method_id(@__retain_semantics Other fileNameExtensionForType:saveOperation:)]
        pub unsafe fn fileNameExtensionForType_saveOperation(
            &self,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
        ) -> Option<Id<NSString>>;

        #[cfg(feature = "NSUserInterfaceValidation")]
        #[method(validateUserInterfaceItem:)]
        pub unsafe fn validateUserInterfaceItem(
            &self,
            item: &ProtocolObject<dyn NSValidatedUserInterfaceItem>,
        ) -> bool;

        #[method(usesUbiquitousStorage)]
        pub unsafe fn usesUbiquitousStorage(mtm: MainThreadMarker) -> bool;

        #[method_id(@__retain_semantics Other presentedItemURL)]
        pub unsafe fn presentedItemURL(&self) -> Option<Id<NSURL>>;

        #[method_id(@__retain_semantics Other observedPresentedItemUbiquityAttributes)]
        pub unsafe fn observedPresentedItemUbiquityAttributes(&self)
            -> Id<NSSet<NSURLResourceKey>>;

        #[cfg(feature = "block2")]
        #[method(relinquishPresentedItemToReader:)]
        pub unsafe fn relinquishPresentedItemToReader(
            &self,
            reader: &Block<dyn Fn(*mut Block<dyn Fn()>)>,
        );

        #[cfg(feature = "block2")]
        #[method(relinquishPresentedItemToWriter:)]
        pub unsafe fn relinquishPresentedItemToWriter(
            &self,
            writer: &Block<dyn Fn(*mut Block<dyn Fn()>)>,
        );

        #[cfg(feature = "block2")]
        #[method(savePresentedItemChangesWithCompletionHandler:)]
        pub unsafe fn savePresentedItemChangesWithCompletionHandler(
            &self,
            completion_handler: &Block<dyn Fn(*mut NSError)>,
        );

        #[cfg(feature = "block2")]
        #[method(accommodatePresentedItemDeletionWithCompletionHandler:)]
        pub unsafe fn accommodatePresentedItemDeletionWithCompletionHandler(
            &self,
            completion_handler: &Block<dyn Fn(*mut NSError)>,
        );

        #[method(presentedItemDidMoveToURL:)]
        pub unsafe fn presentedItemDidMoveToURL(&self, new_url: &NSURL);

        #[method(presentedItemDidChange)]
        pub unsafe fn presentedItemDidChange(&self);

        #[method(presentedItemDidChangeUbiquityAttributes:)]
        pub unsafe fn presentedItemDidChangeUbiquityAttributes(
            &self,
            attributes: &NSSet<NSURLResourceKey>,
        );

        #[method(presentedItemDidGainVersion:)]
        pub unsafe fn presentedItemDidGainVersion(&self, version: &NSFileVersion);

        #[method(presentedItemDidLoseVersion:)]
        pub unsafe fn presentedItemDidLoseVersion(&self, version: &NSFileVersion);

        #[method(presentedItemDidResolveConflictVersion:)]
        pub unsafe fn presentedItemDidResolveConflictVersion(&self, version: &NSFileVersion);
    }
);

extern_methods!(
    /// Methods declared on superclass `NSObject`
    unsafe impl NSDocument {
        #[method_id(@__retain_semantics New new)]
        pub unsafe fn new(mtm: MainThreadMarker) -> Id<Self>;
    }
);

extern_methods!(
    /// NSDeprecated
    unsafe impl NSDocument {
        #[deprecated = "Use -saveToURL:ofType:forSaveOperation:completionHandler: instead"]
        #[method(saveToURL:ofType:forSaveOperation:error:_)]
        pub unsafe fn saveToURL_ofType_forSaveOperation_error(
            &self,
            url: &NSURL,
            type_name: &NSString,
            save_operation: NSSaveOperationType,
        ) -> Result<(), Id<NSError>>;

        #[deprecated]
        #[method_id(@__retain_semantics Other dataRepresentationOfType:)]
        pub unsafe fn dataRepresentationOfType(&self, r#type: &NSString) -> Option<Id<NSData>>;

        #[deprecated]
        #[method_id(@__retain_semantics Other fileAttributesToWriteToFile:ofType:saveOperation:)]
        pub unsafe fn fileAttributesToWriteToFile_ofType_saveOperation(
            &self,
            full_document_path: &NSString,
            document_type_name: &NSString,
            save_operation_type: NSSaveOperationType,
        ) -> Option<Id<NSDictionary>>;

        #[deprecated]
        #[method_id(@__retain_semantics Other fileName)]
        pub unsafe fn fileName(&self) -> Option<Id<NSString>>;

        #[deprecated]
        #[method_id(@__retain_semantics Other fileWrapperRepresentationOfType:)]
        pub unsafe fn fileWrapperRepresentationOfType(
            &self,
            r#type: &NSString,
        ) -> Option<Id<NSFileWrapper>>;

        #[deprecated]
        #[method_id(@__retain_semantics Init initWithContentsOfFile:ofType:)]
        pub unsafe fn initWithContentsOfFile_ofType(
            this: Allocated<Self>,
            absolute_path: &NSString,
            type_name: &NSString,
        ) -> Option<Id<Self>>;

        #[deprecated]
        #[method_id(@__retain_semantics Init initWithContentsOfURL:ofType:)]
        pub unsafe fn initWithContentsOfURL_ofType(
            this: Allocated<Self>,
            url: &NSURL,
            type_name: &NSString,
        ) -> Option<Id<Self>>;

        #[deprecated]
        #[method(loadDataRepresentation:ofType:)]
        pub unsafe fn loadDataRepresentation_ofType(
            &self,
            data: &NSData,
            r#type: &NSString,
        ) -> bool;

        #[deprecated]
        #[method(loadFileWrapperRepresentation:ofType:)]
        pub unsafe fn loadFileWrapperRepresentation_ofType(
            &self,
            wrapper: &NSFileWrapper,
            r#type: &NSString,
        ) -> bool;

        #[deprecated]
        #[method(printShowingPrintPanel:)]
        pub unsafe fn printShowingPrintPanel(&self, flag: bool);

        #[deprecated]
        #[method(readFromFile:ofType:)]
        pub unsafe fn readFromFile_ofType(&self, file_name: &NSString, r#type: &NSString) -> bool;

        #[deprecated]
        #[method(readFromURL:ofType:)]
        pub unsafe fn readFromURL_ofType(&self, url: &NSURL, r#type: &NSString) -> bool;

        #[deprecated]
        #[method(revertToSavedFromFile:ofType:)]
        pub unsafe fn revertToSavedFromFile_ofType(
            &self,
            file_name: &NSString,
            r#type: &NSString,
        ) -> bool;

        #[deprecated]
        #[method(revertToSavedFromURL:ofType:)]
        pub unsafe fn revertToSavedFromURL_ofType(&self, url: &NSURL, r#type: &NSString) -> bool;

        #[cfg(feature = "NSPrintInfo")]
        #[deprecated]
        #[method(runModalPageLayoutWithPrintInfo:)]
        pub unsafe fn runModalPageLayoutWithPrintInfo(&self, print_info: &NSPrintInfo)
            -> NSInteger;

        #[deprecated]
        #[method(saveToFile:saveOperation:delegate:didSaveSelector:contextInfo:)]
        pub unsafe fn saveToFile_saveOperation_delegate_didSaveSelector_contextInfo(
            &self,
            file_name: &NSString,
            save_operation: NSSaveOperationType,
            delegate: Option<&AnyObject>,
            did_save_selector: Option<Sel>,
            context_info: *mut c_void,
        );

        #[deprecated]
        #[method(setFileName:)]
        pub unsafe fn setFileName(&self, file_name: Option<&NSString>);

        #[deprecated]
        #[method(writeToFile:ofType:)]
        pub unsafe fn writeToFile_ofType(&self, file_name: &NSString, r#type: &NSString) -> bool;

        #[deprecated]
        #[method(writeToFile:ofType:originalFile:saveOperation:)]
        pub unsafe fn writeToFile_ofType_originalFile_saveOperation(
            &self,
            full_document_path: &NSString,
            document_type_name: &NSString,
            full_original_document_path: Option<&NSString>,
            save_operation_type: NSSaveOperationType,
        ) -> bool;

        #[deprecated]
        #[method(writeToURL:ofType:)]
        pub unsafe fn writeToURL_ofType(&self, url: &NSURL, r#type: &NSString) -> bool;

        #[deprecated]
        #[method(writeWithBackupToFile:ofType:saveOperation:)]
        pub unsafe fn writeWithBackupToFile_ofType_saveOperation(
            &self,
            full_document_path: &NSString,
            document_type_name: &NSString,
            save_operation_type: NSSaveOperationType,
        ) -> bool;
    }
);
