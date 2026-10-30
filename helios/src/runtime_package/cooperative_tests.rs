use super::*;
use crate::ReplayWorker;
use std::time::Duration;

#[test]
fn cooperative_worker_grids_clamp_along_each_axis() {
    for axis in 0..3 {
        let mut captured = [1; 3];
        let mut expected = [1; 3];
        captured[axis] = 48;
        expected[axis] = 40;
        assert_eq!(clamp_cooperative_grid(captured, 40).unwrap(), expected);
        assert_eq!(clamp_cooperative_grid(expected, 40).unwrap(), expected);
        assert_eq!(clamp_cooperative_grid(captured, 80).unwrap(), captured);
    }
}

#[test]
fn cooperative_grid_clamping_preserves_other_dimensions() {
    for (captured, expected) in [
        ([48, 2, 3], [6, 2, 3]),
        ([2, 48, 3], [2, 6, 3]),
        ([2, 3, 48], [2, 3, 6]),
    ] {
        assert_eq!(clamp_cooperative_grid(captured, 40).unwrap(), expected);
    }
}

#[test]
fn cooperative_grid_clamping_rejects_planes_above_capacity() {
    for (grid, capacity) in [([1, 1, 48], 0), ([8, 8, 8], 40), ([u32::MAX; 3], 40)] {
        assert!(clamp_cooperative_grid(grid, capacity).is_err());
    }
}

#[test]
#[ignore = "requires FLARE_COOPERATIVE_TEST_PACKAGE exported by cooperative_package_fixture.py on the assigned GPU"]
fn oversized_cooperative_grids_replay_at_device_capacity() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLARE_COOPERATIVE_TEST_PACKAGE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let length = expected["length"].as_u64().unwrap() as usize;
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    let poison = vec![0xff; (length * 2 + 3) * 4];
    for _ in 0..4 {
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload(name, &poison).unwrap();
            worker.submit(index, 3).unwrap();
            let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
            assert!(result.error.is_none(), "{:?}", result.error);
            let bytes = owner.download(name).unwrap();
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            for i in 0..length {
                assert_eq!(values[i], i as i32 * 3 + 7, "{name}: element {i}");
                assert_eq!(
                    values[length + i],
                    if name == "ordinary" {
                        -1
                    } else {
                        (length - 1 - i) as i32 * 3 + 18
                    },
                    "{name}: post-barrier element {i}",
                );
            }
            let grid: Vec<i32> = serde_json::from_value(expected["grids"][name].clone()).unwrap();
            assert_eq!(values[length * 2..], grid, "{name}: replayed grid");
        }
    }
    assert!(worker.close().drained);
}

#[test]
#[ignore = "requires FLARE_KERNEL_PREDICATE_TEST_PACKAGE exported by kernel_predicate_package_fixture.py on the assigned GPU"]
fn startup_kernel_predicates_replay_only_the_selected_launch() {
    use crate::execution::GraphExecutor;
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_KERNEL_PREDICATE_TEST_PACKAGE").unwrap());
    for selected in [1i32, 2, 3] {
        let owner = unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
        owner.upload("choice_source", &[selected.to_ne_bytes(), 0i32.to_ne_bytes()].concat()).unwrap();
        let executor = unsafe { GraphExecutor::new(owner.execution_manifest().unwrap()) };
        if selected == 3 {
            assert!(executor.is_err(), "unknown startup output must reject before replay");
            continue;
        }
        let mut executor = executor.unwrap();
        for index in 0..executor.manifest().graphs.len() {
            owner.upload("output", &[0i32, 0, -1, -1].into_iter().flat_map(i32::to_ne_bytes).collect::<Vec<_>>()).unwrap();
            for _ in 0..3 {
                assert!(executor.submit(index, &[], false).unwrap().is_some());
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                loop {
                    if executor.poll().unwrap().is_some() { break; }
                    assert!(std::time::Instant::now() < deadline, "native executor completion timed out");
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            let values: Vec<_> = owner.download("output").unwrap().chunks_exact(4)
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap())).collect();
            let scale = if executor.manifest().graphs[index].name == "decode1" { 1 } else { 2 };
            assert_eq!(values, if selected == 1 { vec![3, 0, scale, -1] } else { vec![0, 3, -1, 2 * scale] });
        }
        assert!(executor.close().drained);
        owner.upload("choice_source", &[3i32.to_ne_bytes(), 0i32.to_ne_bytes()].concat()).unwrap();
        assert!(unsafe { GraphExecutor::new(owner.execution_manifest().unwrap()) }.is_err());
        assert_eq!(owner.download("selector").unwrap(), selected.to_ne_bytes());
        // Inspect the instantiated graph's actual source clone. Losing kernels
        // have been removed, and selected kernels retain their launch policy.
        unsafe {
            let library = crate::os::cuda_library().unwrap();
            let attribute = library.get::<unsafe extern "C" fn(Handle, i32, *mut std::ffi::c_void) -> i32>(
                b"cuGraphKernelNodeGetAttribute\0").unwrap();
            let parameters = library.get::<unsafe extern "C" fn(Handle, *mut driver::KernelParams) -> i32>(
                b"cuGraphKernelNodeGetParams_v2\0").unwrap();
            let nodes = library.get::<unsafe extern "C" fn(Handle, *mut Handle, *mut usize) -> i32>(
                b"cuGraphGetNodes\0").unwrap();
            let kind = library.get::<unsafe extern "C" fn(Handle, *mut i32) -> i32>(b"cuGraphNodeGetType\0").unwrap();
            for graph in owner.execution_manifest().unwrap().graphs {
                let binding = graph.startup_graph.as_ref().unwrap();
                assert!(binding.ready());
                assert_eq!(graph.handle, graph.executable());
                let mut count = 0;
                owner.driver.check("cuGraphGetNodes", nodes(binding.selected_template() as Handle, std::ptr::null_mut(), &mut count)).unwrap();
                assert_eq!(count, 1);
                let mut handles = vec![std::ptr::null_mut(); count];
                owner.driver.check("cuGraphGetNodes", nodes(binding.selected_template() as Handle, handles.as_mut_ptr(), &mut count)).unwrap();
                let mut kernels = 0;
                for node in handles {
                    let mut node_kind = -1;
                    owner.driver.check("cuGraphNodeGetType", kind(node, &mut node_kind)).unwrap();
                    assert_eq!(node_kind, 0);
                    kernels += 1;
                    let mut value = [0u64; 8];
                    owner.driver.check("cuGraphKernelNodeGetAttribute", attribute(node, 2, value.as_mut_ptr().cast())).unwrap();
                    assert_eq!(value[0], u64::from(selected == 2));
                    let mut params: driver::KernelParams = std::mem::zeroed();
                    owner.driver.check("cuGraphKernelNodeGetParams_v2", parameters(node, &mut params)).unwrap();
                    let scale = if graph.name == "decode1" { 1 } else { 2 };
                    assert_eq!(params.grid, [scale * selected as u32, 1, 1]);
                }
                assert_eq!(kernels, 1);
            }
        }
        let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
        for _ in 0..4 {
            for (index, name) in owner.graph_names().iter().enumerate().filter(|(_, name)| name.starts_with("decode")) {
                owner.upload("output", &[0i32, 0, -1, -1].into_iter().flat_map(i32::to_ne_bytes).collect::<Vec<_>>()).unwrap();
                worker.submit(index, 3).unwrap();
                let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
                assert!(result.error.is_none(), "{:?}", result.error);
                let values: Vec<_> = owner.download("output").unwrap().chunks_exact(4)
                    .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap())).collect();
                let scale = if name == "decode1" { 1 } else { 2 };
                assert_eq!(values, if selected == 1 { vec![3, 0, scale, -1] } else { vec![0, 3, -1, 2 * scale] });
            }
        }
        assert!(worker.close().drained);
    }
    let owner = unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    owner.upload("choice_source", &[1i32.to_ne_bytes(), 7i32.to_ne_bytes()].concat()).unwrap();
    assert!(unsafe { GraphExecutor::new(owner.execution_manifest().unwrap()) }.is_err());
}
