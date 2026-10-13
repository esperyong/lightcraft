//! What a selection stands for when collapsed stacks are in it.
//!
//! The grid shows a collapsed stack as one cell, and selecting that cell selects the photo it shows.
//! Photo Merge and the commands that copy develop settings (Sync, Paste, presets, Quick Develop,
//! Auto Sync) take such a photo for its whole stack: the person sees one item standing for the
//! group, e.g. a focus bracket. Ratings, flags, labels, keywords, deleting and exporting keep acting
//! on the photo itself, as before.

use std::collections::HashSet;

use lightcraft_catalog::{MediaKind, PhotoId, StackId};
use serde_json::Value;

use crate::{LibrarySource, Selection, Session};

/// Photos named by the parameters (an `ids` array or a numeric `id`, as [`Session::targets`] reads
/// them; a preset's `id` is no photo).
pub fn explicit_ids(p: &Value) -> bool {
    p.get("ids").and_then(Value::as_array).is_some() || p.get("id").and_then(Value::as_u64).is_some()
}

/// Which members of a collapsed stack a command works on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Members {
    /// Every member that isn't deleted (develop commands).
    All,
    /// The photos to merge: no videos, virtual copies (the same file again) or merge results — a
    /// stack topped by an earlier merge is merged again from its frames. A stack of merge results
    /// only (HDRs for a panorama) keeps them.
    Sources,
}

/// The photos a command works on, with the collapsed stacks they came from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StackScope {
    /// All of them, in order: a photo standing for a collapsed stack is replaced by the stack's
    /// members (in stack order).
    pub ids: Vec<PhotoId>,
    /// The collapsed stacks taken whole, with their members in `ids`.
    pub stacks: Vec<(StackId, Vec<PhotoId>)>,
    /// The photos that stand for themselves.
    pub loose: Vec<PhotoId>,
}

impl StackScope {
    fn literal(ids: Vec<PhotoId>) -> Self {
        StackScope { loose: ids.clone(), ids, stacks: Vec::new() }
    }
    /// Two or more collapsed stacks and nothing else: each is a group of its own (one merge per
    /// focus bracket).
    pub fn per_stack(&self) -> bool {
        self.loose.is_empty() && self.stacks.len() >= 2
    }
    /// The one collapsed stack every photo came from.
    pub fn from_one_stack(&self) -> Option<StackId> {
        match (self.loose.is_empty(), self.stacks.as_slice()) {
            (true, [(id, _)]) => Some(*id),
            _ => None,
        }
    }
    /// The groups to work on one by one: each stack when [`StackScope::per_stack`], else all.
    pub fn groups(&self) -> Vec<Vec<PhotoId>> {
        if self.per_stack() { self.stacks.iter().map(|(_, m)| m.clone()).collect() } else { vec![self.ids.clone()] }
    }
}

impl Session {
    /// `ids` with every photo of a collapsed stack replaced by the stack's `members` (see
    /// [`StackScope`]). In Recently Deleted, where stacks aren't arranged, nothing is expanded.
    pub fn expand_stacks(&self, ids: &[PhotoId], members: Members) -> StackScope {
        if self.source == LibrarySource::RecentlyDeleted || self.catalog.stacks().next().is_none() {
            return StackScope::literal(ids.to_vec());
        }
        // a few photos (the usual selection, asked every frame by the panels): look each up
        let index = (ids.len() > 8).then(|| self.catalog.stack_index());
        let stack_of = |id: &PhotoId| match &index {
            Some(ix) => ix.get(id).and_then(|(sid, _)| self.catalog.stack(*sid)),
            None => self.catalog.stack_of(*id),
        };
        let mut out = StackScope::default();
        let mut seen: HashSet<PhotoId> = HashSet::with_capacity(ids.len());
        let mut done: HashSet<StackId> = HashSet::new();
        for id in ids {
            let stack = stack_of(id).filter(|st| st.collapsed);
            let m = match stack {
                Some(st) if !done.insert(st.id) => continue,
                Some(st) => self.stack_members(&st.photos, members).map(|m| (st.id, m)),
                None => None,
            };
            match m {
                Some((sid, m)) => {
                    out.ids.extend(m.iter().copied().filter(|p| seen.insert(*p)));
                    out.stacks.push((sid, m));
                }
                None if seen.insert(*id) => {
                    out.loose.push(*id);
                    out.ids.push(*id);
                }
                None => {}
            }
        }
        out
    }

    /// The targets of a command (as [`Session::targets`]) with collapsed stacks taken whole. Explicit
    /// `ids` / `id` are taken as given unless `expandStacks: true`.
    pub fn stack_targets(&self, p: &Value, members: Members) -> StackScope {
        let ids = self.targets(p);
        if explicit_ids(p) && !p.get("expandStacks").and_then(Value::as_bool).unwrap_or(false) {
            return StackScope::literal(ids);
        }
        self.expand_stacks(&ids, members)
    }

    /// Whether any of the photos has been edited (its settings differ from what import gave it).
    pub fn any_edited(&self, ids: &[PhotoId]) -> bool {
        ids.iter().any(|id| self.catalog.photo(*id).is_some_and(|p| p.is_edited()))
    }

    /// Keep the active photo visible after stacks changed (collapsed, or new stacks hide it): it
    /// moves to the photo its stack shows.
    pub fn keep_active_visible(&mut self) {
        let vis = self.visible_cloned();
        if let Some(a) = self.selection.active
            && !vis.contains(&a)
            && let Some(st) = self.catalog.stack_of(a)
        {
            let top = vis.iter().copied().find(|p| st.photos.contains(p)).unwrap_or(st.top());
            self.selection = Selection::single(top);
        }
    }

    /// A collapsed stack's `members` (`None` when none is left, e.g. all deleted).
    fn stack_members(&self, photos: &[PhotoId], members: Members) -> Option<Vec<PhotoId>> {
        let live: Vec<_> = photos.iter().filter_map(|id| self.catalog.photo(*id)).filter(|p| !p.deleted).collect();
        let m: Vec<PhotoId> = match members {
            Members::All => live.iter().map(|p| p.id).collect(),
            Members::Sources => {
                let files: Vec<_> = live.into_iter().filter(|p| p.kind != MediaKind::Video && p.copy_of.is_none()).collect();
                let frames: Vec<PhotoId> =
                    files.iter().filter(|p| lightcraft_catalog::query::merged_kind(&p.file_name).is_none()).map(|p| p.id).collect();
                if frames.is_empty() { files.iter().map(|p| p.id).collect() } else { frames }
            }
        };
        (!m.is_empty()).then_some(m)
    }
}
