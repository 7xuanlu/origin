// SPDX-License-Identifier: Apache-2.0
//! Coordination between daemon-owned background writers and one approved repair.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;
use wenlan_types::repair::ApplyRepairRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintenanceFenceError {
    Busy,
    Conflict,
    Expired,
}

impl std::fmt::Display for MaintenanceFenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Busy => "repair_background_writer_busy",
            Self::Conflict => "repair_write_fence_conflict",
            Self::Expired => "repair_write_fence_expired",
        })
    }
}

impl std::error::Error for MaintenanceFenceError {}

#[derive(Debug)]
struct RepairLease {
    manifest_id: String,
    durable: bool,
    active_attempt: Option<RepairAttemptToken>,
}

#[derive(Debug)]
struct PendingRepair {
    manifest_id: String,
    owners: HashSet<PendingRepairToken>,
}

#[derive(Debug)]
struct ReservedRepair {
    predecessor_manifest_id: String,
    next_apply: ApplyRepairRequest,
    expires_at: Instant,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RepairAttemptToken(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PendingRepairToken(u64);

#[derive(Debug, Default)]
struct CoordinatorState {
    recovery_complete: bool,
    active_background: usize,
    active_analysis: usize,
    pending_repair: Option<PendingRepair>,
    repair_lease: Option<RepairLease>,
    // A repair-only daemon is started for one exact approved request. Keep
    // that request for the full process lifetime so a failed apply attempt
    // cannot widen the restricted router into a generic repair executor.
    startup_claim: Option<ApplyRepairRequest>,
    startup_claim_complete: bool,
    reserved_repair: Option<ReservedRepair>,
    // Process-lifetime tombstones prevent a timed-out chained approval from
    // silently falling back to an ordinary apply after background work ran.
    expired_repairs: Vec<ApplyRepairRequest>,
    next_pending_repair: u64,
    next_repair_attempt: u64,
    next_reservation: u64,
}

impl CoordinatorState {
    fn repair_allows_analysis(&self) -> bool {
        self.repair_lease
            .as_ref()
            .is_none_or(|lease| lease.durable && lease.active_attempt.is_none())
    }

    fn issue_pending_repair(&mut self) -> PendingRepairToken {
        self.next_pending_repair = self.next_pending_repair.wrapping_add(1);
        if self.next_pending_repair == 0 {
            self.next_pending_repair = 1;
        }
        PendingRepairToken(self.next_pending_repair)
    }

    fn issue_repair_attempt(&mut self) -> RepairAttemptToken {
        self.next_repair_attempt = self.next_repair_attempt.wrapping_add(1);
        if self.next_repair_attempt == 0 {
            self.next_repair_attempt = 1;
        }
        RepairAttemptToken(self.next_repair_attempt)
    }

    fn issue_reservation(&mut self) -> u64 {
        self.next_reservation = self.next_reservation.wrapping_add(1);
        if self.next_reservation == 0 {
            self.next_reservation = 1;
        }
        self.next_reservation
    }

    fn remove_pending_owner(&mut self, manifest_id: &str, owner: PendingRepairToken) -> bool {
        let (removed, empty) = match self.pending_repair.as_mut() {
            Some(pending) if pending.manifest_id == manifest_id => {
                let removed = pending.owners.remove(&owner);
                (removed, pending.owners.is_empty())
            }
            _ => (false, false),
        };
        if empty {
            self.pending_repair = None;
        }
        removed || empty
    }

    fn clear_pending_manifest(&mut self, manifest_id: &str) {
        if self
            .pending_repair
            .as_ref()
            .is_some_and(|pending| pending.manifest_id == manifest_id)
        {
            self.pending_repair = None;
        }
    }

    fn expire_reservation(&mut self, now: Instant) -> bool {
        let Some(reserved) = self.reserved_repair.as_ref() else {
            return false;
        };
        if reserved.expires_at > now {
            return false;
        }
        let reserved = self
            .reserved_repair
            .take()
            .expect("reservation checked above");
        self.record_expired_repair(reserved.next_apply);
        true
    }

    fn record_expired_repair(&mut self, request: ApplyRepairRequest) {
        if !self.expired_repairs.contains(&request) {
            self.expired_repairs.push(request);
        }
    }

    fn repair_is_expired(&self, request: &ApplyRepairRequest) -> bool {
        self.expired_repairs.contains(request)
    }
}

#[derive(Clone, Debug)]
pub struct MaintenanceCoordinator {
    state: Arc<Mutex<CoordinatorState>>,
    notify: Arc<Notify>,
}

#[derive(Debug)]
struct PendingRepairRegistration {
    coordinator: MaintenanceCoordinator,
    manifest_id: String,
    owner: PendingRepairToken,
    active: bool,
}

impl PendingRepairRegistration {
    fn new(
        coordinator: MaintenanceCoordinator,
        manifest_id: String,
        owner: PendingRepairToken,
    ) -> Self {
        Self {
            coordinator,
            manifest_id,
            owner,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for PendingRepairRegistration {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let mut state = self.coordinator.state.lock().unwrap();
        let changed = state.remove_pending_owner(&self.manifest_id, self.owner);
        drop(state);
        if changed {
            self.coordinator.notify.notify_waiters();
        }
    }
}

impl Default for MaintenanceCoordinator {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(CoordinatorState::default())),
            notify: Arc::new(Notify::new()),
        }
    }
}

impl MaintenanceCoordinator {
    /// Open background maintenance only after durable repair recovery has
    /// completed. The coordinator starts sealed so startup ordering is
    /// enforced by construction rather than by scheduler convention.
    pub fn finish_recovery(&self) {
        let mut state = self.state.lock().unwrap();
        state.recovery_complete = true;
        drop(state);
        self.notify.notify_waiters();
    }

    pub fn try_begin_background(&self) -> Option<BackgroundMaintenanceGuard> {
        let mut state = self.state.lock().unwrap();
        let expired = state.expire_reservation(Instant::now());
        if !state.recovery_complete
            || state.active_analysis != 0
            || state.pending_repair.is_some()
            || state.repair_lease.is_some()
            || state.reserved_repair.is_some()
        {
            drop(state);
            if expired {
                self.notify.notify_waiters();
            }
            return None;
        }
        state.active_background = state.active_background.saturating_add(1);
        drop(state);
        if expired {
            self.notify.notify_waiters();
        }
        Some(BackgroundMaintenanceGuard {
            coordinator: self.clone(),
        })
    }

    /// Wait until startup recovery and any approved repair window finish, then
    /// register one detached daemon writer. The notification is created before
    /// inspecting state so a release cannot be missed between check and await.
    pub async fn begin_background(&self) -> BackgroundMaintenanceGuard {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.state.lock().unwrap();
                let expired = state.expire_reservation(Instant::now());
                if state.recovery_complete
                    && state.active_analysis == 0
                    && state.pending_repair.is_none()
                    && state.repair_lease.is_none()
                    && state.reserved_repair.is_none()
                {
                    state.active_background = state.active_background.saturating_add(1);
                    drop(state);
                    if expired {
                        self.notify.notify_waiters();
                    }
                    return BackgroundMaintenanceGuard {
                        coordinator: self.clone(),
                    };
                }
                drop(state);
                if expired {
                    self.notify.notify_waiters();
                }
            }
            notified.await;
        }
    }

    /// Wait for daemon-owned writers and approved repair work to drain, then
    /// keep background writers out for one diagnostic snapshot.
    pub async fn begin_analysis(&self) -> AnalysisGuard {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.state.lock().unwrap();
                let expired = state.expire_reservation(Instant::now());
                if state.recovery_complete
                    && state.active_background == 0
                    && state.pending_repair.is_none()
                    && state.repair_allows_analysis()
                    && state.reserved_repair.is_none()
                {
                    state.active_analysis = state.active_analysis.saturating_add(1);
                    drop(state);
                    if expired {
                        self.notify.notify_waiters();
                    }
                    return AnalysisGuard {
                        coordinator: self.clone(),
                    };
                }
                drop(state);
                if expired {
                    self.notify.notify_waiters();
                }
            }
            notified.await;
        }
    }

    /// Unapproved-request variant of the repair fence. Production always goes
    /// through `acquire_approved_repair`; this one is kept only so the
    /// state-machine tests still cover the `approved_request: None` branch of
    /// `acquire_repair_inner`.
    #[cfg(test)]
    pub async fn acquire_repair(
        &self,
        manifest_id: &str,
        wait: Duration,
    ) -> Result<RepairFenceGuard, MaintenanceFenceError> {
        self.acquire_repair_inner(manifest_id, None, wait).await
    }

    pub async fn acquire_approved_repair(
        &self,
        request: &ApplyRepairRequest,
        wait: Duration,
    ) -> Result<RepairFenceGuard, MaintenanceFenceError> {
        self.acquire_repair_inner(request.manifest_id(), Some(request), wait)
            .await
    }

    async fn acquire_repair_inner(
        &self,
        manifest_id: &str,
        approved_request: Option<&ApplyRepairRequest>,
        wait: Duration,
    ) -> Result<RepairFenceGuard, MaintenanceFenceError> {
        let deadline = Instant::now() + wait;
        let mut pending_registration: Option<PendingRepairRegistration> = None;
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.state.lock().unwrap();
                let expired = state.expire_reservation(Instant::now());
                if expired {
                    // The mutex still protects the transition; woken tasks
                    // re-check state only after this critical section exits.
                    self.notify.notify_waiters();
                }
                if !state.recovery_complete {
                    return Err(MaintenanceFenceError::Busy);
                }
                if state.startup_claim_complete {
                    return Err(MaintenanceFenceError::Conflict);
                }
                if state
                    .startup_claim
                    .as_ref()
                    .is_some_and(|claim| approved_request != Some(claim))
                {
                    return Err(MaintenanceFenceError::Conflict);
                }
                if approved_request.is_some_and(|request| state.repair_is_expired(request)) {
                    return Err(MaintenanceFenceError::Expired);
                }
                if let Some(reserved) = state.reserved_repair.as_ref() {
                    if approved_request != Some(&reserved.next_apply) {
                        return Err(MaintenanceFenceError::Conflict);
                    }
                    if state.active_background != 0
                        || state.active_analysis != 0
                        || state.repair_lease.is_some()
                    {
                        return Err(MaintenanceFenceError::Busy);
                    }
                    let attempt_token = state.issue_repair_attempt();
                    state.reserved_repair = None;
                    state.repair_lease = Some(RepairLease {
                        manifest_id: manifest_id.to_string(),
                        durable: false,
                        active_attempt: Some(attempt_token),
                    });
                    return Ok(RepairFenceGuard {
                        coordinator: self.clone(),
                        manifest_id: manifest_id.to_string(),
                        attempt_token: Some(attempt_token),
                        retain_on_drop: false,
                    });
                }
                if let Some(lease) = state.repair_lease.as_ref() {
                    if lease.manifest_id != manifest_id {
                        return Err(MaintenanceFenceError::Conflict);
                    }
                    if state.active_analysis != 0
                        || !lease.durable
                        || lease.active_attempt.is_some()
                    {
                        return Err(MaintenanceFenceError::Busy);
                    }
                    let attempt_token = state.issue_repair_attempt();
                    state
                        .repair_lease
                        .as_mut()
                        .expect("repair lease checked above")
                        .active_attempt = Some(attempt_token);
                    if let Some(registration) = pending_registration.as_mut() {
                        state.remove_pending_owner(manifest_id, registration.owner);
                        registration.disarm();
                    }
                    return Ok(RepairFenceGuard {
                        coordinator: self.clone(),
                        manifest_id: manifest_id.to_string(),
                        attempt_token: Some(attempt_token),
                        retain_on_drop: false,
                    });
                }
                match state.pending_repair.as_ref() {
                    Some(pending) if pending.manifest_id != manifest_id => {
                        return Err(MaintenanceFenceError::Conflict);
                    }
                    Some(_) | None => {}
                }
                let owner = match pending_registration.as_ref() {
                    Some(registration) => registration.owner,
                    None => state.issue_pending_repair(),
                };
                match state.pending_repair.as_mut() {
                    Some(pending) => {
                        pending.owners.insert(owner);
                    }
                    None => {
                        state.pending_repair = Some(PendingRepair {
                            manifest_id: manifest_id.to_string(),
                            owners: HashSet::from([owner]),
                        });
                    }
                }
                if pending_registration.is_none() {
                    pending_registration = Some(PendingRepairRegistration::new(
                        self.clone(),
                        manifest_id.to_string(),
                        owner,
                    ));
                }
                if state.active_background == 0 && state.active_analysis == 0 {
                    let attempt_token = state.issue_repair_attempt();
                    state.remove_pending_owner(manifest_id, owner);
                    state.repair_lease = Some(RepairLease {
                        manifest_id: manifest_id.to_string(),
                        durable: false,
                        active_attempt: Some(attempt_token),
                    });
                    pending_registration
                        .as_mut()
                        .expect("pending owner registered before lease acquisition")
                        .disarm();
                    return Ok(RepairFenceGuard {
                        coordinator: self.clone(),
                        manifest_id: manifest_id.to_string(),
                        attempt_token: Some(attempt_token),
                        retain_on_drop: false,
                    });
                }
            }

            let now = Instant::now();
            if now >= deadline
                || tokio::time::timeout(deadline.saturating_duration_since(now), notified)
                    .await
                    .is_err()
            {
                return Err(MaintenanceFenceError::Busy);
            }
        }
    }

    /// Assert the durable fence is held and idle. No production caller —
    /// `acquire_repair_verification` is the shipped path; kept for the fence
    /// state-machine tests.
    #[cfg(test)]
    pub fn require_repair(&self, manifest_id: &str) -> Result<(), MaintenanceFenceError> {
        let state = self.state.lock().unwrap();
        match state.repair_lease.as_ref() {
            Some(lease)
                if lease.manifest_id == manifest_id
                    && lease.durable
                    && lease.active_attempt.is_none() =>
            {
                Ok(())
            }
            Some(lease) if lease.manifest_id == manifest_id => Err(MaintenanceFenceError::Busy),
            Some(_) => Err(MaintenanceFenceError::Conflict),
            None => Err(MaintenanceFenceError::Expired),
        }
    }

    /// Acquire the existing durable fence for one verification attempt. This
    /// closes the check-then-release race with same-manifest apply retries.
    pub fn acquire_repair_verification(
        &self,
        manifest_id: &str,
    ) -> Result<RepairFenceGuard, MaintenanceFenceError> {
        let mut state = self.state.lock().unwrap();
        match state.repair_lease.as_ref() {
            Some(lease) if lease.manifest_id != manifest_id => {
                return Err(MaintenanceFenceError::Conflict)
            }
            Some(lease)
                if state.active_analysis != 0
                    || !lease.durable
                    || lease.active_attempt.is_some() =>
            {
                return Err(MaintenanceFenceError::Busy)
            }
            None => return Err(MaintenanceFenceError::Expired),
            Some(_) => {}
        }
        let attempt_token = state.issue_repair_attempt();
        state
            .repair_lease
            .as_mut()
            .expect("durable repair lease checked above")
            .active_attempt = Some(attempt_token);
        Ok(RepairFenceGuard {
            coordinator: self.clone(),
            manifest_id: manifest_id.to_string(),
            attempt_token: Some(attempt_token),
            retain_on_drop: false,
        })
    }

    pub fn matches_handoff(
        &self,
        predecessor_manifest_id: &str,
        next_apply: &ApplyRepairRequest,
    ) -> bool {
        let mut state = self.state.lock().unwrap();
        let expired = state.expire_reservation(Instant::now());
        let matches = state.reserved_repair.as_ref().is_some_and(|reserved| {
            reserved.predecessor_manifest_id == predecessor_manifest_id
                && reserved.next_apply == *next_apply
        });
        drop(state);
        if expired {
            self.notify.notify_waiters();
        }
        matches
    }

    fn expire_reserved_repair(&self, generation: u64) {
        let mut state = self.state.lock().unwrap();
        let should_expire = state.reserved_repair.as_ref().is_some_and(|reserved| {
            reserved.generation == generation && reserved.expires_at <= Instant::now()
        });
        if !should_expire {
            return;
        }
        let reserved = state
            .reserved_repair
            .take()
            .expect("reservation checked above");
        state.record_expired_repair(reserved.next_apply);
        drop(state);
        self.notify.notify_waiters();
    }

    /// Restore an applied-but-unverified repair fence during daemon startup.
    /// Startup uses `rearm_approved_repair`; this manifest-id-only variant has
    /// no production caller and is kept for the fence state-machine tests.
    #[cfg(test)]
    pub fn rearm_repair(&self, manifest_id: &str) -> Result<(), MaintenanceFenceError> {
        let mut state = self.state.lock().unwrap();
        if state.recovery_complete {
            return Err(MaintenanceFenceError::Conflict);
        }
        if state.startup_claim_complete {
            return Err(MaintenanceFenceError::Conflict);
        }
        if state.active_background != 0 {
            return Err(MaintenanceFenceError::Busy);
        }
        if state.pending_repair.is_some() {
            return Err(MaintenanceFenceError::Conflict);
        }
        if state.reserved_repair.is_some() {
            return Err(MaintenanceFenceError::Conflict);
        }
        if let Some(lease) = state.repair_lease.as_ref() {
            if lease.manifest_id != manifest_id {
                return Err(MaintenanceFenceError::Conflict);
            }
            return Ok(());
        }
        state.repair_lease = Some(RepairLease {
            manifest_id: manifest_id.to_string(),
            durable: true,
            active_attempt: None,
        });
        Ok(())
    }

    /// Arm a repair-only process for one exact user-approved apply request.
    /// Unlike an applied-but-unverified fence, this approval remains exact
    /// across pre-commit failures so no other manifest or digest can use the
    /// restricted execution surface.
    pub fn rearm_approved_repair(
        &self,
        request: ApplyRepairRequest,
    ) -> Result<(), MaintenanceFenceError> {
        let mut state = self.state.lock().unwrap();
        if state.recovery_complete {
            return Err(MaintenanceFenceError::Conflict);
        }
        if state.active_background != 0 {
            return Err(MaintenanceFenceError::Busy);
        }
        if state.pending_repair.is_some() || state.reserved_repair.is_some() {
            return Err(MaintenanceFenceError::Conflict);
        }
        if state
            .startup_claim
            .as_ref()
            .is_some_and(|claim| claim != &request)
        {
            return Err(MaintenanceFenceError::Conflict);
        }
        if let Some(lease) = state.repair_lease.as_ref() {
            if lease.manifest_id != request.manifest_id() {
                return Err(MaintenanceFenceError::Conflict);
            }
            state.startup_claim = Some(request);
            state.startup_claim_complete = false;
            return Ok(());
        }
        state.repair_lease = Some(RepairLease {
            manifest_id: request.manifest_id().to_string(),
            durable: true,
            active_attempt: None,
        });
        state.startup_claim = Some(request);
        state.startup_claim_complete = false;
        Ok(())
    }
}

/// Exclusive ownership of one apply or verification attempt. A newly-created
/// fence is provisional until a durable apply artifact exists. A retry of a
/// durable fence owns only its attempt token: dropping it retains the fence,
/// and no other attempt can retain or release state through that token.
pub struct RepairFenceGuard {
    coordinator: MaintenanceCoordinator,
    manifest_id: String,
    attempt_token: Option<RepairAttemptToken>,
    retain_on_drop: bool,
}

impl RepairFenceGuard {
    /// Fail closed if the request is cancelled after the canonical writer may
    /// have published durable apply state but before the route can inspect it.
    pub fn retain_on_uncertain_drop(&mut self) {
        self.retain_on_drop = true;
    }

    pub fn retain_until_verification(mut self) -> Result<(), MaintenanceFenceError> {
        let attempt_token = self.attempt_token.ok_or(MaintenanceFenceError::Expired)?;
        let mut state = self.coordinator.state.lock().unwrap();
        match state.repair_lease.as_mut() {
            Some(lease)
                if lease.manifest_id == self.manifest_id
                    && lease.active_attempt == Some(attempt_token) =>
            {
                lease.durable = true;
                lease.active_attempt = None;
                state.clear_pending_manifest(&self.manifest_id);
                self.attempt_token = None;
                drop(state);
                self.coordinator.notify.notify_waiters();
                Ok(())
            }
            Some(_) => Err(MaintenanceFenceError::Conflict),
            None => Err(MaintenanceFenceError::Expired),
        }
    }

    /// Release this attempt's fence after the caller has proved that no
    /// durable apply artifact remains. The active-attempt token makes that
    /// decision exclusive: a stale or concurrent retry cannot release the
    /// fence. `Drop` intentionally retains an already-durable fence because it
    /// cannot distinguish a pre-commit failure from uncertain publication.
    pub fn release_after_precommit_failure(mut self) -> Result<(), MaintenanceFenceError> {
        let attempt_token = self.attempt_token.ok_or(MaintenanceFenceError::Expired)?;
        let mut state = self.coordinator.state.lock().unwrap();
        match state.repair_lease.as_ref() {
            Some(lease)
                if lease.manifest_id == self.manifest_id
                    && lease.active_attempt == Some(attempt_token) =>
            {
                if state
                    .startup_claim
                    .as_ref()
                    .is_some_and(|claim| claim.manifest_id() == self.manifest_id)
                {
                    let lease = state
                        .repair_lease
                        .as_mut()
                        .expect("repair lease checked above");
                    lease.durable = true;
                    lease.active_attempt = None;
                } else {
                    state.repair_lease = None;
                    state.clear_pending_manifest(&self.manifest_id);
                }
            }
            Some(_) => return Err(MaintenanceFenceError::Conflict),
            None => return Err(MaintenanceFenceError::Expired),
        }
        self.attempt_token = None;
        drop(state);
        self.coordinator.notify.notify_waiters();
        Ok(())
    }

    /// Release a durable repair after this verification attempt has recorded
    /// its receipt. Only the verification attempt that owns the token can
    /// remove the fence.
    pub fn release_after_verification(mut self) -> Result<(), MaintenanceFenceError> {
        let attempt_token = self.attempt_token.ok_or(MaintenanceFenceError::Expired)?;
        let mut state = self.coordinator.state.lock().unwrap();
        match state.repair_lease.as_ref() {
            Some(lease)
                if lease.manifest_id == self.manifest_id
                    && lease.durable
                    && lease.active_attempt == Some(attempt_token) =>
            {
                state.repair_lease = None;
                state.clear_pending_manifest(&self.manifest_id);
                if state
                    .startup_claim
                    .as_ref()
                    .is_some_and(|claim| claim.manifest_id() == self.manifest_id)
                {
                    state.startup_claim_complete = true;
                }
            }
            Some(lease) if lease.manifest_id == self.manifest_id => {
                return Err(MaintenanceFenceError::Busy)
            }
            Some(_) => return Err(MaintenanceFenceError::Conflict),
            None => return Err(MaintenanceFenceError::Expired),
        }
        self.attempt_token = None;
        drop(state);
        self.coordinator.notify.notify_waiters();
        Ok(())
    }

    /// Atomically replace the verified manifest's durable lease with a
    /// bounded reservation for the next exact approved apply. Background
    /// waiters are intentionally not notified during this handoff.
    pub fn handoff_after_verification(
        mut self,
        next_apply: ApplyRepairRequest,
        ttl: Duration,
    ) -> Result<(), MaintenanceFenceError> {
        if ttl.is_zero() || next_apply.manifest_id() == self.manifest_id {
            return Err(MaintenanceFenceError::Conflict);
        }
        let attempt_token = self.attempt_token.ok_or(MaintenanceFenceError::Expired)?;
        let expires_at = Instant::now() + ttl;
        let mut state = self.coordinator.state.lock().unwrap();
        match state.repair_lease.as_ref() {
            Some(lease)
                if lease.manifest_id == self.manifest_id
                    && lease.durable
                    && lease.active_attempt == Some(attempt_token) => {}
            Some(lease) if lease.manifest_id == self.manifest_id => {
                return Err(MaintenanceFenceError::Busy)
            }
            Some(_) => return Err(MaintenanceFenceError::Conflict),
            None => return Err(MaintenanceFenceError::Expired),
        }
        let generation = state.issue_reservation();
        if state.repair_is_expired(&next_apply) {
            return Err(MaintenanceFenceError::Expired);
        }
        state.repair_lease = None;
        state.clear_pending_manifest(&self.manifest_id);
        if state
            .startup_claim
            .as_ref()
            .is_some_and(|claim| claim.manifest_id() == self.manifest_id)
        {
            state.startup_claim = Some(next_apply.clone());
            state.startup_claim_complete = false;
        }
        state.reserved_repair = Some(ReservedRepair {
            predecessor_manifest_id: self.manifest_id.clone(),
            next_apply,
            expires_at,
            generation,
        });
        self.attempt_token = None;
        drop(state);

        let coordinator = self.coordinator.clone();
        tokio::spawn(async move {
            tokio::time::sleep(ttl).await;
            coordinator.expire_reserved_repair(generation);
        });
        Ok(())
    }
}

impl Drop for RepairFenceGuard {
    fn drop(&mut self) {
        let Some(attempt_token) = self.attempt_token else {
            return;
        };
        let mut state = self.coordinator.state.lock().unwrap();
        match state.repair_lease.as_mut() {
            Some(lease)
                if lease.manifest_id == self.manifest_id
                    && lease.active_attempt == Some(attempt_token) =>
            {
                if lease.durable || self.retain_on_drop {
                    lease.durable = true;
                    lease.active_attempt = None;
                } else {
                    state.repair_lease = None;
                }
            }
            _ => return,
        }
        drop(state);
        self.coordinator.notify.notify_waiters();
    }
}

pub struct BackgroundMaintenanceGuard {
    coordinator: MaintenanceCoordinator,
}

pub struct AnalysisGuard {
    coordinator: MaintenanceCoordinator,
}

impl Drop for AnalysisGuard {
    fn drop(&mut self) {
        let mut state = self.coordinator.state.lock().unwrap();
        state.active_analysis = state.active_analysis.saturating_sub(1);
        drop(state);
        self.coordinator.notify.notify_waiters();
    }
}

impl Drop for BackgroundMaintenanceGuard {
    fn drop(&mut self) {
        let mut state = self.coordinator.state.lock().unwrap();
        state.active_background = state.active_background.saturating_sub(1);
        drop(state);
        self.coordinator.notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use wenlan_types::repair::{ApplyRepairRequest, RepairDigest};

    fn approved(manifest_id: &str, digest_byte: char) -> ApplyRepairRequest {
        let digest = digest_byte.to_string().repeat(64);
        ApplyRepairRequest::try_new(
            manifest_id.to_string(),
            RepairDigest::parse(&digest).unwrap(),
            format!("apply repair {manifest_id} {digest}"),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn repair_waits_for_active_background_job_then_excludes_new_jobs() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator
            .try_begin_background()
            .expect("background starts");
        let waiting = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .acquire_repair("repair_a", Duration::from_secs(1))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());

        drop(background);
        let repair = waiting.await.unwrap().unwrap();
        repair.retain_until_verification().unwrap();
        assert!(coordinator.try_begin_background().is_none());
        assert!(coordinator.require_repair("repair_a").is_ok());
    }

    #[tokio::test]
    async fn analysis_waits_for_background_then_excludes_background_and_repair() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();
        let waiting_analysis = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move { coordinator.begin_analysis().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiting_analysis.is_finished());

        drop(background);
        let analysis = waiting_analysis.await.unwrap();
        assert!(coordinator.try_begin_background().is_none());
        let waiting_repair = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .acquire_repair("repair_a", Duration::from_secs(1))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting_repair.is_finished());

        drop(analysis);
        drop(waiting_repair.await.unwrap().unwrap());
        drop(coordinator.try_begin_background().unwrap());
    }

    #[tokio::test]
    async fn durable_idle_repair_allows_analysis_without_reopening_mutation() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        let mut waiting_analysis = Box::pin(coordinator.begin_analysis());
        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut waiting_analysis)
                .await
                .is_err(),
            "analysis must wait while an apply attempt is active"
        );

        repair.retain_until_verification().unwrap();
        let analysis = tokio::time::timeout(Duration::from_secs(1), &mut waiting_analysis)
            .await
            .expect("retaining a durable idle repair must wake the verification lint");

        assert!(coordinator.try_begin_background().is_none());
        assert!(matches!(
            coordinator.acquire_repair_verification("repair_a"),
            Err(MaintenanceFenceError::Busy)
        ));
        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));

        drop(analysis);
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("verification release resumes background work"),
        );
    }

    #[tokio::test]
    async fn release_resumes_background_jobs() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        repair.retain_until_verification().unwrap();
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("release resumes jobs"),
        );
    }

    #[tokio::test]
    async fn acquire_timeout_clears_pending_repair() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();

        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));
        drop(background);
        drop(
            coordinator
                .try_begin_background()
                .expect("timed-out pending state is cleared"),
        );
    }

    #[tokio::test]
    async fn one_waiter_timeout_does_not_clear_another_waiters_pending_repair() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();

        let waiter_b = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .acquire_repair("repair_a", Duration::from_secs(1))
                    .await
            })
        };
        for _ in 0..100 {
            if coordinator.state.lock().unwrap().pending_repair.is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(coordinator.state.lock().unwrap().pending_repair.is_some());

        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));

        // No task switch occurs between A's timeout and this check. B's
        // pending ownership must therefore still exclude a new background
        // writer even after the original writer drains.
        drop(background);
        assert!(coordinator.try_begin_background().is_none());

        let repair = waiter_b.await.unwrap().unwrap();
        drop(repair);
        drop(
            coordinator
                .try_begin_background()
                .expect("background resumes after the remaining waiter releases"),
        );
    }

    #[tokio::test]
    async fn provisional_handoff_preserves_an_unpolled_waiters_pending_ownership() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();
        let mut waiter_a = Box::pin(coordinator.acquire_repair("repair_a", Duration::from_secs(1)));
        let mut waiter_b = Box::pin(coordinator.acquire_repair("repair_a", Duration::from_secs(1)));

        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut waiter_a)
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut waiter_b)
                .await
                .is_err()
        );

        drop(background);
        let provisional = tokio::time::timeout(Duration::from_secs(1), &mut waiter_a)
            .await
            .expect("first waiter wakes after the background writer drains")
            .expect("first waiter acquires the provisional fence");
        drop(provisional);

        assert!(
            coordinator.try_begin_background().is_none(),
            "the still-live unpolled waiter must retain pending ownership"
        );
        drop(waiter_b);
        drop(
            coordinator
                .try_begin_background()
                .expect("cancelling the last waiter reopens background work"),
        );
    }

    #[tokio::test]
    async fn durable_apply_clears_stale_pending_waiters_before_verification_release() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();
        let mut waiter_a = Box::pin(coordinator.acquire_repair("repair_a", Duration::from_secs(1)));
        let mut waiter_b = Box::pin(coordinator.acquire_repair("repair_a", Duration::from_secs(1)));
        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut waiter_a)
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(5), &mut waiter_b)
                .await
                .is_err()
        );

        drop(background);
        let repair = tokio::time::timeout(Duration::from_secs(1), &mut waiter_a)
            .await
            .unwrap()
            .unwrap();
        repair.retain_until_verification().unwrap();
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .release_after_verification()
            .unwrap();

        drop(
            coordinator
                .try_begin_background()
                .expect("an unpolled stale waiter cannot resurrect after verification"),
        );
        drop(waiter_b);
    }

    #[tokio::test]
    async fn cancelled_waiter_releases_only_its_pending_ownership() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let background = coordinator.try_begin_background().unwrap();

        let waiter = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move {
                coordinator
                    .acquire_repair("repair_a", Duration::from_secs(1))
                    .await
            })
        };
        for _ in 0..100 {
            if coordinator.state.lock().unwrap().pending_repair.is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(coordinator.state.lock().unwrap().pending_repair.is_some());

        waiter.abort();
        match waiter.await {
            Err(join_error) => assert!(join_error.is_cancelled()),
            Ok(_) => panic!("aborted repair waiter unexpectedly completed"),
        }
        drop(background);
        drop(
            coordinator
                .try_begin_background()
                .expect("cancelling the sole waiter clears its pending ownership"),
        );
    }

    #[test]
    fn background_stays_sealed_until_startup_recovery_finishes() {
        let coordinator = MaintenanceCoordinator::default();
        assert!(coordinator.try_begin_background().is_none());

        coordinator.finish_recovery();
        drop(
            coordinator
                .try_begin_background()
                .expect("background opens only after recovery"),
        );
    }

    #[tokio::test]
    async fn applied_unverified_repair_fence_does_not_expire() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        repair.retain_until_verification().unwrap();

        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(coordinator.try_begin_background().is_none());
        assert!(coordinator.require_repair("repair_a").is_ok());
    }

    #[tokio::test]
    async fn concurrent_apply_for_same_manifest_cannot_share_durable_fence() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();

        let retry = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();

        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_secs(1))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));
        assert!(matches!(
            coordinator.acquire_repair_verification("repair_a"),
            Err(MaintenanceFenceError::Busy)
        ));

        retry.release_after_precommit_failure().unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("the owning retry may release after proving no durable artifact remains"),
        );
    }

    #[tokio::test]
    async fn dropping_durable_retry_relinquishes_attempt_but_retains_repair_fence() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();

        let retry = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        drop(retry);

        assert!(coordinator.require_repair("repair_a").is_ok());
        assert!(coordinator.try_begin_background().is_none());
        let next_retry = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .expect("dropping an attempt releases ownership, not the durable fence");
        next_retry.retain_until_verification().unwrap();
    }

    #[tokio::test]
    async fn verification_owns_durable_fence_against_apply_retries() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();

        let verification = coordinator
            .acquire_repair_verification("repair_a")
            .expect("verification acquires the idle durable fence");

        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_secs(1))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));
        verification.release_after_verification().unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("verified owner releases the durable fence"),
        );
    }

    #[tokio::test]
    async fn verification_handoff_blocks_background_until_next_manifest_finishes() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair_a = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        repair_a.retain_until_verification().unwrap();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');

        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_secs(1))
            .unwrap();

        assert!(coordinator.try_begin_background().is_none());
        let repair_b = coordinator
            .acquire_approved_repair(&next_apply, Duration::from_secs(1))
            .await
            .expect("the exact reserved apply consumes the handoff");
        repair_b.retain_until_verification().unwrap();
        assert!(coordinator.try_begin_background().is_none());
        coordinator
            .acquire_repair_verification(next_apply.manifest_id())
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("terminal verification resumes maintenance"),
        );
    }

    #[tokio::test]
    async fn queued_background_waiter_is_not_woken_between_manifests() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let waiting = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move { coordinator.begin_background().await })
        };
        tokio::task::yield_now().await;
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_secs(1))
            .unwrap();
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());

        let repair_b = coordinator
            .acquire_approved_repair(&next_apply, Duration::from_secs(1))
            .await
            .unwrap();
        repair_b.retain_until_verification().unwrap();
        assert!(!waiting.is_finished());
        coordinator
            .acquire_repair_verification(next_apply.manifest_id())
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(waiting.await.unwrap());
    }

    #[tokio::test]
    async fn reserved_next_manifest_is_exact_and_exclusive() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        let wrong_apply = approved("repair_550e8400-e29b-41d4-a716-446655440002", 'c');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_secs(1))
            .unwrap();

        assert!(matches!(
            coordinator
                .acquire_approved_repair(&wrong_apply, Duration::from_secs(1))
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        let repair_b = coordinator
            .acquire_approved_repair(&next_apply, Duration::from_secs(1))
            .await
            .unwrap();
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&next_apply, Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));
        drop(repair_b);
    }

    #[tokio::test]
    async fn expired_handoff_rejects_the_chained_apply_and_resumes_background() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_millis(5))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&next_apply, Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Expired)
        ));
        drop(
            coordinator
                .try_begin_background()
                .expect("expired handoff resumes maintenance"),
        );
    }

    #[tokio::test]
    async fn expired_handoff_tombstone_survives_unrelated_handoffs() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let expired_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(expired_apply.clone(), Duration::from_millis(5))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(15)).await;

        let repair_c = coordinator
            .acquire_repair("repair_c", Duration::from_secs(1))
            .await
            .unwrap();
        repair_c.retain_until_verification().unwrap();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440002", 'c');
        coordinator
            .acquire_repair_verification("repair_c")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_secs(1))
            .unwrap();
        coordinator
            .acquire_approved_repair(&next_apply, Duration::from_secs(1))
            .await
            .unwrap()
            .retain_until_verification()
            .unwrap();
        coordinator
            .acquire_repair_verification(next_apply.manifest_id())
            .unwrap()
            .release_after_verification()
            .unwrap();

        assert!(matches!(
            coordinator
                .acquire_approved_repair(&expired_apply, Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Expired)
        ));
    }

    #[tokio::test]
    async fn lazy_handoff_expiry_wakes_background_waiter() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_secs(60))
            .unwrap();
        let waiting = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move { coordinator.begin_background().await })
        };
        tokio::task::yield_now().await;
        coordinator
            .state
            .lock()
            .unwrap()
            .reserved_repair
            .as_mut()
            .unwrap()
            .expires_at = Instant::now() - Duration::from_millis(1);

        assert!(matches!(
            coordinator
                .acquire_approved_repair(&next_apply, Duration::from_millis(5))
                .await,
            Err(MaintenanceFenceError::Expired)
        ));
        drop(
            tokio::time::timeout(Duration::from_secs(1), waiting)
                .await
                .expect("lazy expiry wakes the queued background waiter")
                .unwrap(),
        );
    }

    #[tokio::test]
    async fn stale_handoff_expiry_cannot_clear_the_claimed_next_lease() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let next_apply = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .handoff_after_verification(next_apply.clone(), Duration::from_millis(5))
            .unwrap();
        coordinator
            .acquire_approved_repair(&next_apply, Duration::from_secs(1))
            .await
            .unwrap()
            .retain_until_verification()
            .unwrap();

        tokio::time::sleep(Duration::from_millis(15)).await;
        assert!(coordinator.try_begin_background().is_none());
        assert!(coordinator.require_repair(next_apply.manifest_id()).is_ok());
    }

    #[tokio::test]
    async fn dropping_an_uncertain_apply_attempt_retains_the_fence() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let mut repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        repair.retain_on_uncertain_drop();
        drop(repair);

        assert!(coordinator.try_begin_background().is_none());
        assert!(coordinator.require_repair("repair_a").is_ok());
    }

    #[tokio::test]
    async fn concurrent_apply_for_same_manifest_cannot_share_provisional_fence() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();

        assert!(matches!(
            coordinator
                .acquire_repair("repair_a", Duration::from_secs(1))
                .await,
            Err(MaintenanceFenceError::Busy)
        ));
        assert_eq!(
            coordinator.require_repair("repair_a"),
            Err(MaintenanceFenceError::Busy)
        );
        drop(repair);
        drop(
            coordinator
                .try_begin_background()
                .expect("failed apply releases only its provisional fence"),
        );
    }

    #[tokio::test]
    async fn detached_writer_waits_until_durable_repair_is_verified() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.finish_recovery();
        let repair = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();
        repair.retain_until_verification().unwrap();
        let waiting = {
            let coordinator = coordinator.clone();
            tokio::spawn(async move { coordinator.begin_background().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());

        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(waiting.await.unwrap());
    }

    #[test]
    fn restart_rearm_excludes_background_until_verification_releases_it() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();

        assert!(coordinator.try_begin_background().is_none());
        assert!(coordinator.require_repair("repair_a").is_ok());
        assert_eq!(
            coordinator.rearm_repair("repair_b"),
            Err(MaintenanceFenceError::Conflict)
        );

        coordinator
            .acquire_repair_verification("repair_a")
            .unwrap()
            .release_after_verification()
            .unwrap();
        drop(
            coordinator
                .try_begin_background()
                .expect("verification resumes jobs"),
        );
    }

    #[tokio::test]
    async fn retried_precommit_failure_releases_rearmed_fence() {
        let coordinator = MaintenanceCoordinator::default();
        coordinator.rearm_repair("repair_a").unwrap();
        coordinator.finish_recovery();
        let retry = coordinator
            .acquire_repair("repair_a", Duration::from_secs(1))
            .await
            .unwrap();

        retry.release_after_precommit_failure().unwrap();

        drop(
            coordinator
                .try_begin_background()
                .expect("a retry with no durable artifact must resume background jobs"),
        );
    }

    #[tokio::test]
    async fn startup_claim_remains_exact_after_precommit_failure() {
        let coordinator = MaintenanceCoordinator::default();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let exact = approved(manifest_id, 'a');
        let wrong_digest = approved(manifest_id, 'b');
        let other_manifest = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'c');
        coordinator.rearm_approved_repair(exact.clone()).unwrap();
        coordinator.finish_recovery();

        let attempt = coordinator
            .acquire_approved_repair(&exact, Duration::from_secs(1))
            .await
            .unwrap();
        attempt.release_after_precommit_failure().unwrap();

        assert!(matches!(
            coordinator
                .acquire_approved_repair(&wrong_digest, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&other_manifest, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        assert!(matches!(
            coordinator
                .acquire_repair(manifest_id, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        coordinator
            .acquire_approved_repair(&exact, Duration::from_secs(1))
            .await
            .expect("the exact startup approval remains retryable");
    }

    #[tokio::test]
    async fn verified_startup_claim_seals_apply_but_releases_the_writer_fence() {
        let coordinator = MaintenanceCoordinator::default();
        let exact = approved("repair_550e8400-e29b-41d4-a716-446655440000", 'a');
        let other = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        coordinator.rearm_approved_repair(exact.clone()).unwrap();
        coordinator.finish_recovery();

        coordinator
            .acquire_approved_repair(&exact, Duration::from_secs(1))
            .await
            .unwrap()
            .retain_until_verification()
            .unwrap();
        coordinator
            .acquire_repair_verification(exact.manifest_id())
            .unwrap()
            .release_after_verification()
            .unwrap();

        drop(
            coordinator
                .try_begin_background()
                .expect("verified startup claim releases the process fence"),
        );
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&exact, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&other, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
    }

    #[tokio::test]
    async fn startup_claim_handoff_rebinds_exact_authority_to_the_successor() {
        let coordinator = MaintenanceCoordinator::default();
        let first = approved("repair_550e8400-e29b-41d4-a716-446655440000", 'a');
        let next = approved("repair_550e8400-e29b-41d4-a716-446655440001", 'b');
        let other = approved("repair_550e8400-e29b-41d4-a716-446655440002", 'c');
        coordinator.rearm_approved_repair(first.clone()).unwrap();
        coordinator.finish_recovery();

        coordinator
            .acquire_approved_repair(&first, Duration::from_secs(1))
            .await
            .unwrap()
            .retain_until_verification()
            .unwrap();
        coordinator
            .acquire_repair_verification(first.manifest_id())
            .unwrap()
            .handoff_after_verification(next.clone(), Duration::from_secs(1))
            .unwrap();

        assert!(matches!(
            coordinator
                .acquire_approved_repair(&first, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&other, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        coordinator
            .acquire_approved_repair(&next, Duration::from_secs(1))
            .await
            .unwrap()
            .release_after_precommit_failure()
            .unwrap();
        assert!(matches!(
            coordinator
                .acquire_approved_repair(&other, Duration::ZERO)
                .await,
            Err(MaintenanceFenceError::Conflict)
        ));
        coordinator
            .acquire_approved_repair(&next, Duration::from_secs(1))
            .await
            .expect("the exact successor remains retryable");
    }
}
