//! Startup construction of a selected CUDA graph with explicit executable ownership.

use std::sync::{Arc, atomic::{AtomicU8, AtomicUsize, Ordering}};
use crate::{Result, cuda::Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphExecutableOwnership {
    /// The package resource owner destroys the selected executable after drain.
    Native,
    /// Python retains the captured executable; the executor owns its selected clone.
    Borrowed,
}

#[derive(Debug)]
pub struct GraphExecutableBinding {
    pub template: usize,
    pub flags: u64,
    pub ownership: GraphExecutableOwnership,
    handle: AtomicUsize,
    selected_template: AtomicUsize,
    // 0: captured, 1: constructing, 2: ready, 3: failed or released.
    state: AtomicU8,
}

impl GraphExecutableBinding {
    pub fn new(template: usize, handle: usize, flags: u64, ownership: GraphExecutableOwnership) -> Arc<Self> {
        Arc::new(Self { template, flags, ownership, handle: AtomicUsize::new(handle),
            selected_template: AtomicUsize::new(0), state: AtomicU8::new(0) })
    }

    pub fn handle(&self) -> usize { self.handle.load(Ordering::Acquire) }
    pub fn selected_template(&self) -> usize { self.selected_template.load(Ordering::Acquire) }
    pub fn ready(&self) -> bool { self.state.load(Ordering::Acquire) == 2 }
    pub(crate) fn unprepared(&self) -> bool { self.state.load(Ordering::Acquire) == 0 }
    pub(crate) fn take_handle(&self) -> usize { self.handle.swap(0, Ordering::AcqRel) }
    pub(crate) fn take_template(&self) -> usize { self.selected_template.swap(0, Ordering::AcqRel) }

    pub(crate) fn prepare(&self, stream: &Stream, disabled: &[usize]) -> Result<()> {
        self.state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "captured graph startup construction may run only once")?;
        let result = (|| {
            let before = stream.free_device_memory()?;
            let graph = stream.clone_pruned_graph(self.template, disabled)?;
            self.selected_template.store(graph, Ordering::Release);
            // Clear the shared slot before destruction: startup failure must not expose
            // the retired executable or destroy it twice during owner teardown.
            let previous = self.take_handle();
            if self.ownership == GraphExecutableOwnership::Native {
                stream.destroy_graph_executable(previous)?;
            }
            let executable = stream.instantiate_graph(graph, self.flags)?;
            self.handle.store(executable, Ordering::Release);
            stream.upload_graph(executable)?;
            stream.synchronize()?;
            if self.ownership == GraphExecutableOwnership::Native {
                let after = stream.free_device_memory()?;
                if after < before {
                    return Err(format!("selected graph exceeds its reserved executable memory by {} bytes", before - after).into());
                }
            }
            Ok(())
        })();
        self.state.store(if result.is_ok() { 2 } else { 3 }, Ordering::Release);
        result
    }

    pub(crate) fn release_borrowed(&self, stream: &Stream) -> Result<()> {
        if self.ownership != GraphExecutableOwnership::Borrowed || self.selected_template() == 0 { return Ok(()); }
        let executable = self.take_handle();
        if executable != 0 { stream.destroy_graph_executable(executable)?; }
        let template = self.take_template();
        if template != 0 { stream.destroy_graph_template(template)?; }
        self.state.store(3, Ordering::Release);
        Ok(())
    }
}
