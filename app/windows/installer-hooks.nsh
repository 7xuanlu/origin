; Wenlan NSIS installer hooks (tauri.windows.conf.json, bundle.windows.nsis.installerHooks).
;
; Tauri's per-user default install directory is %LOCALAPPDATA%\Wenlan. The CLI
; data root is %LOCALAPPDATA%\wenlan, and Windows paths are case-insensitive,
; so the app was installed inside its own data (first-run gauntlet finding F6):
; the uninstaller could never remove its directory, and a future "delete my
; data" flow could take the app with it. Move only that default under
; Programs; a directory the user picked in the installer is kept.
!macro NSIS_HOOK_PREINSTALL
  ${If} $INSTDIR == "$LOCALAPPDATA\${PRODUCTNAME}"
    StrCpy $INSTDIR "$LOCALAPPDATA\Programs\${PRODUCTNAME}"
    SetOutPath $INSTDIR
  ${EndIf}
!macroend
