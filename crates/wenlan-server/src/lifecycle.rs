// SPDX-License-Identifier: Apache-2.0
//! Cooperative daemon shutdown shared by HTTP and background workers.

use std::time::Duration;
use tokio::sync::watch;

#[derive(Clone)]
pub struct ShutdownHandle {
    sender: watch::Sender<bool>,
}

impl Default for ShutdownHandle {
    fn default() -> Self {
        let (sender, _receiver) = watch::channel(false);
        Self { sender }
    }
}

impl ShutdownHandle {
    pub fn request(&self) {
        self.sender.send_replace(true);
    }

    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.sender.subscribe()
    }

    pub fn is_requested(&self) -> bool {
        *self.sender.borrow()
    }
}

pub async fn wait_for_shutdown(mut receiver: watch::Receiver<bool>) {
    while !*receiver.borrow() {
        if receiver.changed().await.is_err() {
            break;
        }
    }
}

pub fn shutdown_requested(receiver: &watch::Receiver<bool>) -> bool {
    *receiver.borrow()
}

/// Returns true when shutdown won the race, false when the sleep completed.
pub async fn sleep_or_shutdown(receiver: &mut watch::Receiver<bool>, duration: Duration) -> bool {
    if shutdown_requested(receiver) {
        return true;
    }
    tokio::select! {
        _ = tokio::time::sleep(duration) => false,
        result = receiver.changed() => result.is_err() || shutdown_requested(receiver),
    }
}
