// SPDX-License-Identifier: AGPL-3.0-only
if (
  import.meta.env.DEV &&
  import.meta.env.VITE_ENABLE_REACT_DEVTOOLS === "1" &&
  import.meta.env.VITE_DISABLE_REACT_DEVTOOLS !== "1"
) {
  void import("react-grab");
  void import("react-scan");
}
