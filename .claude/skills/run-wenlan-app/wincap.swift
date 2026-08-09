// Capture the largest wenlan-app window to a PNG, even when occluded.
// Usage: swift wincap.swift /path/out.png
// Requires Screen Recording permission for the invoking terminal.
import ScreenCaptureKit
import CoreGraphics
import Foundation
import AppKit

let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/wenlan-shot.png"
_ = NSApplication.shared // initialize window-server connection (else CGS_REQUIRE_INIT abort)
let sem = DispatchSemaphore(value: 0)
Task {
  do {
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    // The app owns a secondary ~500x500 window; sort by area to get the main one.
    let wins = content.windows.filter {
      ($0.owningApplication?.applicationName.lowercased().contains("wenlan") ?? false)
    }.sorted { $0.frame.width * $0.frame.height > $1.frame.width * $1.frame.height }
    guard let win = wins.first else { print("NO WINDOW"); exit(1) }
    print("window id=\(win.windowID) frame=\(win.frame) pid=\(win.owningApplication?.processID ?? 0)")
    let filter = SCContentFilter(desktopIndependentWindow: win)
    let cfg = SCStreamConfiguration()
    cfg.width = Int(win.frame.width) * 2 // retina
    cfg.height = Int(win.frame.height) * 2
    cfg.showsCursor = false
    let img = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: cfg)
    let rep = NSBitmapImageRep(cgImage: img)
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: outPath))
    print("saved \(outPath)")
  } catch {
    print("ERR:", error)
    exit(1)
  }
  sem.signal()
}
sem.wait()
