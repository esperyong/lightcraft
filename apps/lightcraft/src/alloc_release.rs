//! Give memory the system allocator keeps after frees back to the system
//! ([`lightcraft_engine::memory::set_release_hook`]). The platform call lives in
//! `lightcraft-sysmem`, one of the workspace's two crates allowed `unsafe`.

/// Install the hook.
pub fn install() {
    lightcraft_engine::memory::set_release_hook(release);
}

fn release() {
    let _ = lightcraft_sysmem::release_free_memory();
}
