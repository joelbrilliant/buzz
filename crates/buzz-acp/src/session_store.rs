//! Durable channel → ACP session bindings for harness restarts.
//!
//! `SessionState` is in-memory only. Agents that advertise `loadSession` (e.g.
//! Hermes) can restore a prior ACP conversation after the harness respawns if
//! the channel→session mapping survives. This module persists that mapping as
//! a small JSON sidecar under the process data directory.
//!
//! Keyed by `(agent_command_identity, agent_args, channel_id)` so different
//! agent binaries / profiles do not share bindings. Heartbeats are never
//! stored — they stay ephemeral.
//!
//! Cross-process safety: the v1 binding map remains backward-readable, while
//! restore intent is written to a separate per-binding guard that legacy
//! whole-file rewrites cannot erase. Corrected processes also retain a
//! per-binding OS lease for the active channel lifetime. Already-running older
//! binaries do not honor either mechanism and must be stopped before cutover;
//! rolling mixed-version safety cannot be enforced by this module.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::config::normalize_agent_command_identity;

/// Environment override for the session store path (tests / operators).
pub const SESSION_STORE_ENV: &str = "BUZZ_ACP_SESSION_STORE";

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
struct StoreFile {
    /// version for future migrations
    version: u32,
    /// map key → ACP session id
    sessions: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RestoreGuardFile {
    version: u32,
    state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
}

const RESTORE_GUARD_VERSION: u32 = 1;

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoredSessionBinding {
    Bound(String),
    Indeterminate(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RestoreClaim {
    NoBinding,
    Claimed(String),
    Indeterminate(String),
    Unavailable,
}

#[derive(Debug, PartialEq, Eq)]
enum RestoreGuard {
    Missing,
    Creating,
    Ready(String),
    Restoring(String),
}

/// Durable session binding store shared across buzz-acp processes.
pub struct SessionStore {
    path: PathBuf,
    lock_path: PathBuf,
    channel_leases: Mutex<HashMap<String, StoreLock>>,
    volatile_blocks: Mutex<HashSet<String>>,
}

/// RAII wrapper that unlocks the OS file lock on drop.
struct StoreLock {
    file: File,
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

impl SessionStore {
    /// Open or create the store at the resolved path.
    ///
    /// Does not cache file contents; each operation reloads under lock.
    pub fn open(path: PathBuf) -> Self {
        // Pin relative BUZZ_ACP_SESSION_STORE overrides to this process's
        // current directory up front. A bare filename otherwise has an empty
        // parent, which cannot be opened for the final durability sync.
        let path = if path.is_absolute() {
            path
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        };
        let lock_path = sibling_lock_path(&path);
        Self {
            path,
            lock_path,
            channel_leases: Mutex::new(HashMap::new()),
            volatile_blocks: Mutex::new(HashSet::new()),
        }
    }

    /// Atomically reserve a channel for this process and record restore intent
    /// before any stateful ACP wire request is allowed.
    ///
    /// The process-local map retains an OS lock for the lifetime of this store,
    /// fencing same-version peers even after a successful load clears the
    /// durable in-progress marker. A lock/read/serialization/commit failure is
    /// distinct from a genuinely absent binding and therefore fails closed.
    pub(crate) fn claim_restore(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
    ) -> RestoreClaim {
        let key = binding_key(agent_command, agent_args, channel_id);
        if self.is_key_blocked(&key) {
            return RestoreClaim::Unavailable;
        }
        if !self.retain_channel_lease(&key) {
            self.block_key(&key);
            return RestoreClaim::Unavailable;
        }

        let Some(_lock) = self.acquire_lock(true) else {
            self.block_key(&key);
            return RestoreClaim::Unavailable;
        };
        let data = match load_store(&self.path) {
            Ok(data) => data,
            Err(e) => {
                self.warn_io("failed to read ACP session bindings before restore", &e);
                self.block_key(&key);
                return RestoreClaim::Unavailable;
            }
        };
        let guard = match read_restore_guard(&self.path, &key) {
            Ok(guard) => guard,
            Err(e) => {
                self.warn_io("failed to read ACP session restore guard", &e);
                self.block_key(&key);
                return RestoreClaim::Unavailable;
            }
        };
        let Some(session_id) = data.sessions.get(&key).cloned() else {
            return match guard {
                RestoreGuard::Missing => {
                    if let Err(e) = persist_creating_guard(&self.path, &key) {
                        self.warn_io("failed to persist ACP session creation intent", &e);
                        self.block_key(&key);
                        RestoreClaim::Unavailable
                    } else {
                        RestoreClaim::NoBinding
                    }
                }
                RestoreGuard::Creating | RestoreGuard::Ready(_) | RestoreGuard::Restoring(_) => {
                    RestoreClaim::Unavailable
                }
            };
        };
        match guard {
            RestoreGuard::Restoring(ref guarded) if guarded == &session_id => {
                return RestoreClaim::Indeterminate(session_id);
            }
            RestoreGuard::Ready(ref guarded) if guarded == &session_id => {
                if let Err(e) = replace_restore_guard(
                    &self.path,
                    &key,
                    &RestoreGuard::Ready(session_id.clone()),
                    "restoring",
                    Some(&session_id),
                ) {
                    self.warn_io("failed to persist ACP session restore intent", &e);
                    self.block_key(&key);
                    return RestoreClaim::Unavailable;
                }
                return RestoreClaim::Claimed(session_id);
            }
            RestoreGuard::Creating | RestoreGuard::Ready(_) | RestoreGuard::Restoring(_) => {
                self.block_key(&key);
                return RestoreClaim::Unavailable;
            }
            RestoreGuard::Missing => {}
        }

        if let Err(e) = persist_restore_guard(&self.path, &key, &session_id) {
            self.warn_io("failed to persist ACP session restore intent", &e);
            self.block_key(&key);
            return RestoreClaim::Unavailable;
        }
        RestoreClaim::Claimed(session_id)
    }

    /// Commit a successful restore by atomically transitioning the exact
    /// write-ahead guard to `Ready`. Either the old `Restoring` state or the
    /// new `Ready` state is safe across a crash; no guard deletion is used.
    /// Failure blocks publication for the current process.
    pub(crate) fn confirm_restore(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
        expected_session_id: &str,
    ) -> bool {
        let key = binding_key(agent_command, agent_args, channel_id);
        let Some(_lock) = self.acquire_lock(true) else {
            self.block_key(&key);
            return false;
        };
        let data = match load_store(&self.path) {
            Ok(data) => data,
            Err(e) => {
                self.warn_io("failed to read ACP session bindings after restore", &e);
                self.block_key(&key);
                return false;
            }
        };
        let binding_matches = data
            .sessions
            .get(&key)
            .is_some_and(|current| current == expected_session_id);
        if !binding_matches {
            self.block_key(&key);
            return false;
        }
        let guard_matches = matches!(
            read_restore_guard(&self.path, &key),
            Ok(RestoreGuard::Restoring(ref current)) if current == expected_session_id
        );
        if !guard_matches {
            self.block_key(&key);
            return false;
        }
        if let Err(e) = replace_restore_guard(
            &self.path,
            &key,
            &RestoreGuard::Restoring(expected_session_id.to_owned()),
            "ready",
            Some(expected_session_id),
        ) {
            self.warn_io("failed to commit ACP session restore", &e);
            self.block_key(&key);
            return false;
        }
        true
    }

    pub(crate) fn retains_channel_lease(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
    ) -> bool {
        let key = binding_key(agent_command, agent_args, channel_id);
        let leases = match self.channel_leases.lock() {
            Ok(leases) => leases,
            Err(poisoned) => poisoned.into_inner(),
        };
        let retained = leases.contains_key(&key);
        drop(leases);
        retained && !self.is_key_blocked(&key)
    }

    pub(crate) fn block_channel(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
    ) {
        self.block_key(&binding_key(agent_command, agent_args, channel_id));
    }

    fn is_key_blocked(&self, key: &str) -> bool {
        let blocks = match self.volatile_blocks.lock() {
            Ok(blocks) => blocks,
            Err(poisoned) => poisoned.into_inner(),
        };
        blocks.contains(key)
    }

    fn block_key(&self, key: &str) {
        let mut blocks = match self.volatile_blocks.lock() {
            Ok(blocks) => blocks,
            Err(poisoned) => poisoned.into_inner(),
        };
        blocks.insert(key.to_owned());
    }

    fn retain_channel_lease(&self, key: &str) -> bool {
        let mut leases = match self.channel_leases.lock() {
            Ok(leases) => leases,
            Err(poisoned) => poisoned.into_inner(),
        };
        if leases.contains_key(key) {
            return true;
        }

        let lease_path = channel_lease_path(&self.path, key);
        if let Some(parent) = lease_path.parent() {
            if let Err(e) = create_dir_all_durable(parent) {
                self.warn_io("failed to commit ACP session lease directory", &e);
                return false;
            }
        }
        let file = match OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lease_path)
        {
            Ok(file) => file,
            Err(e) => {
                self.warn_io("failed to open ACP session channel lease", &e);
                return false;
            }
        };
        if let Err(e) = FileExt::try_lock_exclusive(&file) {
            self.warn_io("ACP session channel is owned by another process", &e);
            return false;
        }
        leases.insert(key.to_owned(), StoreLock { file });
        true
    }

    /// Resolve the default store path for this agent identity.
    pub fn default_path(agent_command: &str, agent_args: &[String]) -> PathBuf {
        if let Ok(override_path) = std::env::var(SESSION_STORE_ENV) {
            if !override_path.trim().is_empty() {
                return PathBuf::from(override_path);
            }
        }
        let identity = store_identity(agent_command, agent_args);
        let base = dirs::data_local_dir()
            .or_else(dirs::data_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("buzz-acp")
            .join("sessions")
            .join(format!("{identity}.json"))
    }

    /// Look up the stored ACP session binding and its durable restore state.
    #[cfg(test)]
    pub(crate) fn get_binding(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
    ) -> Option<StoredSessionBinding> {
        let key = binding_key(agent_command, agent_args, channel_id);
        let _lock = self.acquire_lock(false)?;
        match load_store(&self.path) {
            Ok(data) => {
                let session_id = data.sessions.get(&key)?.clone();
                match read_restore_guard(&self.path, &key) {
                    Ok(RestoreGuard::Restoring(guarded)) if guarded == session_id => {
                        Some(StoredSessionBinding::Indeterminate(session_id))
                    }
                    Ok(RestoreGuard::Missing) => Some(StoredSessionBinding::Bound(session_id)),
                    Ok(RestoreGuard::Ready(guarded)) if guarded == session_id => {
                        Some(StoredSessionBinding::Bound(session_id))
                    }
                    Ok(
                        RestoreGuard::Creating
                        | RestoreGuard::Ready(_)
                        | RestoreGuard::Restoring(_),
                    )
                    | Err(_) => None,
                }
            }
            Err(e) => {
                self.warn_io("failed to read ACP session bindings", &e);
                None
            }
        }
    }

    /// Look up a stored ACP session id without discarding its durable state.
    /// Callers that decide whether to load must use [`Self::get_binding`].
    #[cfg(test)]
    pub fn get(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
    ) -> Option<String> {
        self.get_binding(agent_command, agent_args, channel_id)
            .map(|binding| match binding {
                StoredSessionBinding::Bound(id) | StoredSessionBinding::Indeterminate(id) => id,
            })
    }

    /// Commit a newly-created session binding before any session publication.
    pub(crate) fn commit_new_binding(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
        session_id: &str,
    ) -> bool {
        let key = binding_key(agent_command, agent_args, channel_id);
        if self.is_key_blocked(&key)
            || !self.retains_channel_lease(agent_command, agent_args, channel_id)
        {
            self.block_key(&key);
            return false;
        }
        let Some(_lock) = self.acquire_lock(true) else {
            self.block_key(&key);
            return false;
        };
        let mut data = match load_store(&self.path) {
            Ok(data) => data,
            Err(e) => {
                self.warn_io(
                    "failed to read ACP session bindings before creation commit",
                    &e,
                );
                self.block_key(&key);
                return false;
            }
        };
        if !matches!(
            read_restore_guard(&self.path, &key),
            Ok(RestoreGuard::Creating)
        ) {
            self.block_key(&key);
            return false;
        }
        data.version = 1;
        data.sessions.insert(key.clone(), session_id.to_owned());
        if let Err(e) = save_store(&self.path, &data) {
            self.warn_io("failed to persist ACP session binding", &e);
            self.block_key(&key);
            return false;
        }
        if let Err(e) = replace_restore_guard(
            &self.path,
            &key,
            &RestoreGuard::Creating,
            "ready",
            Some(session_id),
        ) {
            self.warn_io("failed to commit ACP session creation guard", &e);
            self.block_key(&key);
            return false;
        }
        true
    }

    /// Test-only direct binding writer for storage/CAS fixtures.
    #[cfg(test)]
    pub fn put(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
        session_id: &str,
    ) {
        let key = binding_key(agent_command, agent_args, channel_id);
        let _lock = self.acquire_lock(true).expect("test store lock");
        let mut data = load_store(&self.path).unwrap_or_default();
        data.version = 1;
        data.sessions.insert(key.clone(), session_id.to_owned());
        save_store(&self.path, &data).expect("test binding save");
        let _ = remove_restore_guard(&self.path, &key);
    }

    /// Quarantine a binding only if it still points at `expected_session_id`.
    ///
    /// The compare-and-set keeps an old process from quarantining a fresher
    /// binding written by another process. Quarantine is durable so a harness
    /// restart cannot silently retry an ambiguous stateful load.
    #[cfg(test)]
    pub fn mark_indeterminate_if_equals(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
        expected_session_id: &str,
    ) -> bool {
        let key = binding_key(agent_command, agent_args, channel_id);
        let Some(_lock) = self.acquire_lock(true) else {
            return false;
        };
        match load_store(&self.path) {
            Ok(data) => {
                let matches = data
                    .sessions
                    .get(&key)
                    .is_some_and(|current| current == expected_session_id);
                if !matches {
                    return false;
                }
                if let Err(e) = persist_restore_guard(&self.path, &key, expected_session_id) {
                    if e.kind() == std::io::ErrorKind::AlreadyExists
                        && matches!(
                            read_restore_guard(&self.path, &key),
                            Ok(RestoreGuard::Restoring(ref current)) if current == expected_session_id
                        )
                    {
                        return true;
                    }
                    self.warn_io("failed to persist indeterminate ACP session binding", &e);
                    return false;
                }
                true
            }
            Err(e) => {
                self.warn_io("failed to read ACP session bindings before quarantine", &e);
                false
            }
        }
    }

    /// Remove a binding only if it still points at `expected_session_id`.
    ///
    /// Retained as a test-only primitive for compare-and-set coverage; failed
    /// loads are quarantined rather than removed.
    ///
    /// Returns `true` when a matching binding was removed.
    #[cfg(test)]
    pub fn remove_if_equals(
        &self,
        agent_command: &str,
        agent_args: &[String],
        channel_id: &Uuid,
        expected_session_id: &str,
    ) -> bool {
        let key = binding_key(agent_command, agent_args, channel_id);
        let Some(_lock) = self.acquire_lock(true) else {
            return false;
        };
        match load_store(&self.path) {
            Ok(mut data) => {
                let matches = data
                    .sessions
                    .get(&key)
                    .is_some_and(|current| current == expected_session_id);
                if !matches {
                    return false;
                }
                data.sessions.remove(&key);
                if let Err(e) = save_store(&self.path, &data) {
                    self.warn_io(
                        "failed to persist conditional ACP session binding removal",
                        &e,
                    );
                    return false;
                }
                if let Err(e) = remove_restore_guard(&self.path, &key) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        self.warn_io(
                            "failed to remove ACP session restore guard with binding",
                            &e,
                        );
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                self.warn_io(
                    "failed to read ACP session bindings before conditional removal",
                    &e,
                );
                false
            }
        }
    }

    fn acquire_lock(&self, exclusive: bool) -> Option<StoreLock> {
        if let Some(parent) = self.lock_path.parent() {
            if let Err(e) = create_dir_all_durable(parent) {
                self.warn_io("failed to durably create ACP session store directory", &e);
                return None;
            }
        }
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.lock_path)
        {
            Ok(file) => file,
            Err(e) => {
                self.warn_io("failed to open ACP session store lock", &e);
                return None;
            }
        };
        let result = if exclusive {
            FileExt::lock_exclusive(&file)
        } else {
            FileExt::lock_shared(&file)
        };
        if let Err(e) = result {
            self.warn_io("failed to lock ACP session store", &e);
            return None;
        }
        Some(StoreLock { file })
    }

    fn warn_io(&self, message: &'static str, error: &std::io::Error) {
        tracing::warn!(
            target: "session_store",
            path = %self.path.display(),
            lock_path = %self.lock_path.display(),
            error = %error,
            "{message}"
        );
    }
}

fn sibling_lock_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(OsString::from(".lock"));
    PathBuf::from(name)
}

fn binding_digest(binding_key: &str) -> String {
    hex::encode(Sha256::digest(binding_key.as_bytes()))
}

fn restore_state_dir(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(OsString::from(".restore"));
    PathBuf::from(name)
}

fn channel_lease_path(path: &Path, binding_key: &str) -> PathBuf {
    restore_state_dir(path).join(format!("{}.lock", binding_digest(binding_key)))
}

fn restore_guard_path(path: &Path, binding_key: &str) -> PathBuf {
    restore_state_dir(path).join(format!("{}.json", binding_digest(binding_key)))
}

fn read_restore_guard(path: &Path, binding_key: &str) -> std::io::Result<RestoreGuard> {
    match fs::read_to_string(restore_guard_path(path, binding_key)) {
        Ok(text) => {
            let guard: RestoreGuardFile = serde_json::from_str(&text)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            if guard.version != RESTORE_GUARD_VERSION {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unsupported ACP session restore guard",
                ));
            }
            match (guard.state.as_str(), guard.session_id) {
                ("creating", None) => Ok(RestoreGuard::Creating),
                ("ready", Some(session_id)) if !session_id.is_empty() => {
                    Ok(RestoreGuard::Ready(session_id))
                }
                ("restoring", Some(session_id)) if !session_id.is_empty() => {
                    Ok(RestoreGuard::Restoring(session_id))
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid ACP session restore guard state",
                )),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(RestoreGuard::Missing),
        Err(e) => Err(e),
    }
}

fn persist_creating_guard(path: &Path, binding_key: &str) -> std::io::Result<()> {
    persist_guard(path, binding_key, "creating", None)
}

fn persist_restore_guard(path: &Path, binding_key: &str, session_id: &str) -> std::io::Result<()> {
    persist_guard(path, binding_key, "restoring", Some(session_id))
}

fn persist_guard(
    path: &Path,
    binding_key: &str,
    state: &str,
    session_id: Option<&str>,
) -> std::io::Result<()> {
    let guard_path = restore_guard_path(path, binding_key);
    if let Some(parent) = guard_path.parent() {
        create_dir_all_durable(parent)?;
    }
    let encoded = serde_json::to_vec(&RestoreGuardFile {
        version: RESTORE_GUARD_VERSION,
        state: state.to_owned(),
        session_id: session_id.map(str::to_owned),
    })
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&guard_path)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    let parent = guard_path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "guard path has no parent")
    })?;
    sync_directory(parent)
}

fn replace_restore_guard(
    path: &Path,
    binding_key: &str,
    expected: &RestoreGuard,
    state: &str,
    session_id: Option<&str>,
) -> std::io::Result<()> {
    if &read_restore_guard(path, binding_key)? != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ACP session restore guard changed before commit",
        ));
    }
    let guard_path = restore_guard_path(path, binding_key);
    let parent = guard_path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "guard path has no parent")
    })?;
    let tmp = guard_path.with_extension("json.tmp");
    let encoded = serde_json::to_vec(&RestoreGuardFile {
        version: RESTORE_GUARD_VERSION,
        state: state.to_owned(),
        session_id: session_id.map(str::to_owned),
    })
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, &guard_path)?;
    sync_directory(parent)
}

#[cfg(test)]
fn remove_restore_guard(path: &Path, binding_key: &str) -> std::io::Result<()> {
    let guard_path = restore_guard_path(path, binding_key);
    fs::remove_file(&guard_path)?;
    let parent = guard_path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "guard path has no parent")
    })?;
    sync_directory(parent)
}

fn create_dir_all_durable(path: &Path) -> std::io::Result<()> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut cursor = Some(path.as_path());
    while let Some(directory) = cursor {
        if directory.exists() {
            break;
        }
        missing.push(directory.to_owned());
        cursor = directory.parent();
    }
    fs::create_dir_all(&path)?;
    // Commit every newly-created directory entry from the highest missing
    // ancestor down. Syncing only the leaf parent can lose an entire nested
    // BUZZ_ACP_SESSION_STORE subtree after an acknowledged first-run commit.
    for directory in missing.iter().rev() {
        let parent = directory.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "new session-store directory has no parent",
            )
        })?;
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

    let mut options = OpenOptions::new();
    #[cfg(windows)]
    options.read(true).write(true);
    #[cfg(not(windows))]
    options.read(true);
    #[cfg(windows)]
    options.custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    options.open(path)?.sync_all()
}

fn store_identity(agent_command: &str, agent_args: &[String]) -> String {
    let cmd = normalize_agent_command_identity(agent_command);
    let args = agent_args.join(" ");
    let raw = if args.is_empty() {
        cmd
    } else {
        format!("{cmd} {args}")
    };
    // Keep the filename filesystem-safe and short.
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "agent".into()
    } else {
        out
    }
}

fn binding_key(agent_command: &str, agent_args: &[String], channel_id: &Uuid) -> String {
    format!(
        "{}|{}|{}",
        normalize_agent_command_identity(agent_command),
        agent_args.join("\u{1f}"),
        channel_id
    )
}

fn load_store(path: &Path) -> std::io::Result<StoreFile> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(StoreFile::default()),
        Err(e) => Err(e),
    }
}

fn save_store(path: &Path, data: &StoreFile) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all_durable(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "store path has no parent")
    })?;
    sync_directory(parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn first_run_nested_store_creates_complete_durable_state() {
        let dir = tempdir().unwrap();
        let path = dir
            .path()
            .join("nested")
            .join("operator")
            .join("sessions.json");
        let channel = Uuid::new_v4();
        let args = vec!["chat".to_string()];
        let key = binding_key("hermes", &args, &channel);
        let store = SessionStore::open(path.clone());

        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::NoBinding
        ));
        assert!(store.commit_new_binding("hermes", &args, &channel, "sess-nested"));
        assert!(path.exists());
        assert!(matches!(
            read_restore_guard(&path, &key),
            Ok(RestoreGuard::Ready(ref id)) if id == "sess-nested"
        ));
    }

    #[test]
    fn bare_relative_store_commits_first_session_without_empty_parent() {
        let filename = format!(".buzz-session-store-test-{}.json", Uuid::new_v4());
        let channel = Uuid::new_v4();
        let args = vec!["chat".to_string()];
        let store = SessionStore::open(PathBuf::from(&filename));
        let absolute_path = store.path.clone();

        assert!(absolute_path.is_absolute());
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::NoBinding
        ));
        assert!(store.commit_new_binding("hermes", &args, &channel, "sess-relative"));
        assert!(absolute_path.exists());

        let lock_path = store.lock_path.clone();
        let restore_dir = restore_state_dir(&absolute_path);
        drop(store);
        fs::remove_file(absolute_path).unwrap();
        fs::remove_file(lock_path).unwrap();
        fs::remove_dir_all(restore_dir).unwrap();
    }

    #[test]
    fn round_trip_binding() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let store = SessionStore::open(path);
        let channel = Uuid::new_v4();
        assert!(store.get("hermes", &["acp".into()], &channel).is_none());
        store.put("hermes", &["acp".into()], &channel, "sess-1");
        assert_eq!(
            store.get("hermes", &["acp".into()], &channel).as_deref(),
            Some("sess-1")
        );
        // Re-open from disk.
        let store2 = SessionStore::open(store.path.clone());
        assert_eq!(
            store2.get("hermes", &["acp".into()], &channel).as_deref(),
            Some("sess-1")
        );
        assert!(store2.remove_if_equals("hermes", &["acp".into()], &channel, "sess-1"));
        assert!(store2.get("hermes", &["acp".into()], &channel).is_none());
    }

    #[test]
    fn indeterminate_binding_survives_reopen_and_blocks_implicit_reload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let channel = Uuid::new_v4();
        let args = ["acp".into()];
        let store = SessionStore::open(path.clone());

        store.put("hermes", &args, &channel, "sess-live");
        assert!(store.mark_indeterminate_if_equals("hermes", &args, &channel, "sess-live"));

        let reopened = SessionStore::open(path);
        assert!(matches!(
            reopened.get_binding("hermes", &args, &channel),
            Some(StoredSessionBinding::Indeterminate(ref id)) if id == "sess-live"
        ));
        assert!(!reopened.mark_indeterminate_if_equals(
            "hermes",
            &args,
            &channel,
            "different-session"
        ));

        // Reconciliation is explicit: remove the quarantined mapping and guard
        // before writing a replacement. A normal put never erases a guard.
        assert!(reopened.remove_if_equals("hermes", &args, &channel, "sess-live"));
        reopened.put("hermes", &args, &channel, "sess-reconciled");
        assert!(matches!(
            reopened.get_binding("hermes", &args, &channel),
            Some(StoredSessionBinding::Bound(ref id)) if id == "sess-reconciled"
        ));
    }

    #[test]
    fn restore_claim_is_durable_before_wire_and_exclusive_across_store_instances() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let channel = Uuid::new_v4();
        let args = ["acp".into()];
        let process_a = SessionStore::open(path.clone());
        let process_b = SessionStore::open(path.clone());

        process_a.put("hermes", &args, &channel, "sess-live");
        assert!(matches!(
            process_a.claim_restore("hermes", &args, &channel),
            RestoreClaim::Claimed(ref id) if id == "sess-live"
        ));

        // The durable marker is committed before any ACP request is allowed.
        let observer = SessionStore::open(path.clone());
        assert!(matches!(
            observer.get_binding("hermes", &args, &channel),
            Some(StoredSessionBinding::Indeterminate(ref id)) if id == "sess-live"
        ));

        // A peer cannot race a second restore while A owns this channel.
        assert!(matches!(
            process_b.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));

        // Confirming a successful load clears only the durable intent. The
        // process-local OS lease remains held for the active channel session.
        assert!(process_a.confirm_restore("hermes", &args, &channel, "sess-live"));
        assert!(matches!(
            observer.get_binding("hermes", &args, &channel),
            Some(StoredSessionBinding::Bound(ref id)) if id == "sess-live"
        ));
        assert!(matches!(
            process_b.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));

        drop(process_a);
        // A process that observed a competing owner remains fail-closed. A
        // fresh process may claim only after the prior OS lease is released.
        assert!(matches!(
            process_b.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));
        let process_c = SessionStore::open(path.clone());
        assert!(matches!(
            process_c.claim_restore("hermes", &args, &channel),
            RestoreClaim::Claimed(ref id) if id == "sess-live"
        ));

        let empty_channel = Uuid::new_v4();
        let empty_a = SessionStore::open(path.clone());
        let empty_b = SessionStore::open(path);
        assert!(matches!(
            empty_a.claim_restore("hermes", &args, &empty_channel),
            RestoreClaim::NoBinding
        ));
        assert!(matches!(
            empty_b.claim_restore("hermes", &args, &empty_channel),
            RestoreClaim::Unavailable
        ));
    }

    #[test]
    fn creation_intent_is_durable_before_wire_and_commit_precedes_publication() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let channel = Uuid::new_v4();
        let args = ["acp".into()];
        let store = SessionStore::open(path.clone());

        // NoBinding is returned only after a durable Creating guard exists.
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::NoBinding
        ));
        let key = binding_key("hermes", &args, &channel);
        assert!(matches!(
            read_restore_guard(&path, &key),
            Ok(RestoreGuard::Creating)
        ));

        // A confirmed session/new is publishable only after the binding file
        // is durable and the Creating guard is atomically transitioned to Ready.
        assert!(store.commit_new_binding("hermes", &args, &channel, "sess-created"));
        assert!(matches!(
            store.get_binding("hermes", &args, &channel),
            Some(StoredSessionBinding::Bound(ref id)) if id == "sess-created"
        ));
        assert!(matches!(
            read_restore_guard(&path, &key),
            Ok(RestoreGuard::Ready(ref id)) if id == "sess-created"
        ));
    }

    #[test]
    fn failed_creation_commit_blocks_same_process_and_restart() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let channel = Uuid::new_v4();
        let args = ["acp".into()];
        let store = SessionStore::open(path.clone());
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::NoBinding
        ));

        // Force the main binding commit to fail after the write-ahead guard,
        // as if session/new had already returned an exact session id.
        fs::create_dir(&path).unwrap();
        assert!(!store.commit_new_binding("hermes", &args, &channel, "sess-uncommitted"));
        fs::remove_dir(&path).unwrap();
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));
        drop(store);

        // Repairing/restarting does not turn the missing main binding into
        // permission for another session/new; the Creating guard survives.
        let reopened = SessionStore::open(path);
        assert!(matches!(
            reopened.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));
    }

    #[test]
    fn restore_claim_storage_failures_never_become_no_binding() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let channel = Uuid::new_v4();
        let args = ["acp".into()];
        let store = SessionStore::open(path.clone());
        store.put("hermes", &args, &channel, "sess-live");

        // Make the per-binding guard unreadable. A storage failure must not be
        // collapsed into absence or permission for an ACP wire request.
        let key = binding_key("hermes", &args, &channel);
        fs::create_dir_all(restore_guard_path(&path, &key)).unwrap();
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));
        assert_eq!(
            load_store(&path)
                .unwrap()
                .sessions
                .get(&key)
                .map(String::as_str),
            Some("sess-live")
        );
        fs::remove_dir(restore_guard_path(&path, &key)).unwrap();
        assert!(matches!(
            store.claim_restore("hermes", &args, &channel),
            RestoreClaim::Unavailable
        ));

        let corrupt_path = dir.path().join("corrupt.json");
        fs::write(&corrupt_path, "{not-json").unwrap();
        let corrupt = SessionStore::open(corrupt_path);
        assert!(matches!(
            corrupt.claim_restore("hermes", &args, &Uuid::new_v4()),
            RestoreClaim::Unavailable
        ));

        let unsupported_path = dir.path().join("unsupported.json");
        let unsupported_channel = Uuid::new_v4();
        let unsupported = SessionStore::open(unsupported_path.clone());
        unsupported.put("hermes", &args, &unsupported_channel, "sess-versioned");
        let unsupported_key = binding_key("hermes", &args, &unsupported_channel);
        let guard_path = restore_guard_path(&unsupported_path, &unsupported_key);
        fs::create_dir_all(guard_path.parent().unwrap()).unwrap();
        fs::write(
            guard_path,
            r#"{"version":99,"session_id":"sess-versioned"}"#,
        )
        .unwrap();
        assert!(matches!(
            unsupported.claim_restore("hermes", &args, &unsupported_channel),
            RestoreClaim::Unavailable
        ));
    }

    #[test]
    fn legacy_main_store_rewrite_cannot_erase_separate_restore_guard() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let args = ["acp".into()];
        let guarded_channel = Uuid::new_v4();
        let unrelated_channel = Uuid::new_v4();
        let mismatch_channel = Uuid::new_v4();
        let store = SessionStore::open(path.clone());

        store.put("hermes", &args, &guarded_channel, "session-x");
        store.put("hermes", &args, &mismatch_channel, "session-old");
        assert!(store.mark_indeterminate_if_equals("hermes", &args, &guarded_channel, "session-x"));
        assert!(store.mark_indeterminate_if_equals(
            "hermes",
            &args,
            &mismatch_channel,
            "session-old"
        ));

        // Simulate the base/v1 writer rewriting the complete main JSON map.
        // Separate restore guards are outside that writer's schema and survive.
        let _lock = store.acquire_lock(true).unwrap();
        let mut legacy = load_store(&path).unwrap();
        legacy.version = 1;
        legacy.sessions.insert(
            binding_key("hermes", &args, &unrelated_channel),
            "session-other".into(),
        );
        legacy.sessions.insert(
            binding_key("hermes", &args, &mismatch_channel),
            "session-new".into(),
        );
        save_store(&path, &legacy).unwrap();
        drop(_lock);

        let corrected = SessionStore::open(path.clone());
        assert!(matches!(
            corrected.claim_restore("hermes", &args, &guarded_channel),
            RestoreClaim::Indeterminate(ref id) if id == "session-x"
        ));
        let conflict = SessionStore::open(path);
        assert!(matches!(
            conflict.claim_restore("hermes", &args, &mismatch_channel),
            RestoreClaim::Unavailable
        ));
    }

    #[test]
    fn different_args_are_isolated() {
        let dir = tempdir().unwrap();
        let store = SessionStore::open(dir.path().join("s.json"));
        let channel = Uuid::new_v4();
        store.put("hermes", &["acp".into()], &channel, "a");
        store.put(
            "hermes",
            &["-p".into(), "chad".into(), "acp".into()],
            &channel,
            "b",
        );
        assert_eq!(
            store.get("hermes", &["acp".into()], &channel).as_deref(),
            Some("a")
        );
        assert_eq!(
            store
                .get(
                    "hermes",
                    &["-p".into(), "chad".into(), "acp".into()],
                    &channel
                )
                .as_deref(),
            Some("b")
        );
    }

    #[test]
    fn independently_opened_stores_do_not_lose_updates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let store_a = SessionStore::open(path.clone());
        let store_b = SessionStore::open(path.clone());
        let channel_a = Uuid::new_v4();
        let channel_b = Uuid::new_v4();
        let channel_c = Uuid::new_v4();
        let args = ["acp".into()];

        store_a.put("hermes", &args, &channel_a, "session-a");
        store_b.put("hermes", &args, &channel_b, "session-b");

        let reopened = SessionStore::open(path.clone());
        assert_eq!(
            reopened.get("hermes", &args, &channel_a).as_deref(),
            Some("session-a")
        );
        assert_eq!(
            reopened.get("hermes", &args, &channel_b).as_deref(),
            Some("session-b")
        );

        // Open both before either mutation. A stale process-local snapshot would
        // resurrect channel A when the second store writes channel C.
        let remover = SessionStore::open(path.clone());
        let writer = SessionStore::open(path.clone());
        assert!(remover.remove_if_equals("hermes", &args, &channel_a, "session-a"));
        writer.put("hermes", &args, &channel_c, "session-c");

        let final_store = SessionStore::open(path);
        assert!(final_store.get("hermes", &args, &channel_a).is_none());
        assert_eq!(
            final_store.get("hermes", &args, &channel_b).as_deref(),
            Some("session-b")
        );
        assert_eq!(
            final_store.get("hermes", &args, &channel_c).as_deref(),
            Some("session-c")
        );
    }

    #[test]
    fn put_recovers_from_corrupt_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        fs::write(&path, "{not-json").unwrap();
        let store = SessionStore::open(path.clone());
        let channel = Uuid::new_v4();
        store.put("hermes", &["acp".into()], &channel, "recovered");
        assert_eq!(
            store.get("hermes", &["acp".into()], &channel).as_deref(),
            Some("recovered")
        );
    }

    #[test]
    fn remove_if_equals_does_not_delete_newer_binding() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let args = ["acp".into()];
        let channel = Uuid::new_v4();

        // Process A reads X.
        let process_a = SessionStore::open(path.clone());
        process_a.put("hermes", &args, &channel, "session-x");
        let read_x = process_a
            .get("hermes", &args, &channel)
            .expect("process A read X");
        assert_eq!(read_x, "session-x");

        // Process B writes Y for the same channel.
        let process_b = SessionStore::open(path.clone());
        process_b.put("hermes", &args, &channel, "session-y");
        assert_eq!(
            process_b.get("hermes", &args, &channel).as_deref(),
            Some("session-y")
        );

        // Process A's failed load of X must not delete Y.
        let removed = process_a.remove_if_equals("hermes", &args, &channel, &read_x);
        assert!(!removed);

        let final_store = SessionStore::open(path);
        assert_eq!(
            final_store.get("hermes", &args, &channel).as_deref(),
            Some("session-y")
        );
    }

    #[test]
    fn remove_if_equals_clears_matching_stale_binding() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let args = ["acp".into()];
        let channel = Uuid::new_v4();
        let store = SessionStore::open(path.clone());
        store.put("hermes", &args, &channel, "session-x");
        assert!(store.remove_if_equals("hermes", &args, &channel, "session-x"));
        assert!(store.get("hermes", &args, &channel).is_none());
        // No-op when already gone.
        assert!(!store.remove_if_equals("hermes", &args, &channel, "session-x"));
    }
}
