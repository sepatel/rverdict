//! Picks a backend at runtime, proves it computes correctly with a tiny
//! model, and falls back toward CPU when it does not.

use std::panic::{self, AssertUnwindSafe};
use std::str::FromStr;

use burn::backend::flex::FlexDevice;
use burn::{Dispatch, DispatchDevice};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use rverdict_model::{EncoderConfig, OptionAttention, PackedSequence, build_input};

#[cfg(any(
    all(feature = "vulkan", feature = "metal"),
    all(feature = "vulkan", feature = "webgpu"),
    all(feature = "metal", feature = "webgpu"),
))]
compile_error!("enable only one of the `vulkan`, `metal` and `webgpu` features");

/// What the caller asked for. Every choice falls back to CPU if its device is
/// missing or fails the self-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendChoice {
    /// CUDA, then ROCm, then a discrete or integrated GPU through wgpu, then CPU.
    #[default]
    Auto,
    Cpu,
    /// wgpu's default adapter, whatever it is (including software drivers).
    Wgpu,
    Cuda,
    Rocm,
}

impl FromStr for BackendChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" | "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            "rocm" => Ok(Self::Rocm),
            other => Err(format!(
                "unknown backend {other:?}: expected auto, cpu, wgpu, cuda or rocm"
            )),
        }
    }
}

impl BackendChoice {
    /// `RVERDICT_BACKEND`, or `Auto` when unset.
    pub fn from_env() -> Result<Self, String> {
        std::env::var("RVERDICT_BACKEND").map_or(Ok(Self::Auto), |v| v.parse())
    }
}

/// The backend the engine runs on, and why any preferred one was skipped.
#[derive(Debug, Clone)]
pub struct SelectedBackend {
    pub device: DispatchDevice,
    pub name: String,
    pub skipped: Vec<(String, String)>,
}

fn cpu() -> (String, DispatchDevice) {
    ("cpu (flex)".to_owned(), DispatchDevice::Flex(FlexDevice))
}

#[cfg(any(feature = "vulkan", feature = "metal", feature = "webgpu"))]
mod wgpu_adapters {
    use burn::DispatchDevice;
    use burn::backend::wgpu::WgpuDevice;
    use wgpu::DeviceType;

    #[cfg(feature = "vulkan")]
    const API: &str = "vulkan";
    #[cfg(feature = "metal")]
    const API: &str = "metal";
    #[cfg(feature = "webgpu")]
    const API: &str = "webgpu";

    fn dispatch(device: WgpuDevice) -> DispatchDevice {
        #[cfg(feature = "vulkan")]
        return DispatchDevice::Vulkan(device);
        #[cfg(feature = "metal")]
        return DispatchDevice::Metal(device);
        #[cfg(feature = "webgpu")]
        return DispatchDevice::Wgpu(device);
    }

    /// The adapters cubecl will see, in its order, for the graphics API it
    /// compiles for. WebGPU's automatic API is Metal on macOS, Vulkan elsewhere.
    fn adapters() -> Vec<(DeviceType, String)> {
        let metal =
            cfg!(feature = "metal") || (cfg!(feature = "webgpu") && cfg!(target_os = "macos"));
        let backends = if metal {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        pollster::block_on(instance.enumerate_adapters(backends))
            .iter()
            .map(|adapter| {
                let info = adapter.get_info();
                (info.device_type, info.name)
            })
            .collect()
    }

    /// Hardware GPUs, discrete before integrated, as cubecl indexes them.
    pub(super) fn hardware() -> Vec<(String, DispatchDevice)> {
        let adapters = adapters();
        let of_type = |wanted: DeviceType, make: fn(usize) -> WgpuDevice| {
            adapters
                .iter()
                .filter(move |(kind, _)| *kind == wanted)
                .enumerate()
                .map(move |(i, (_, name))| (format!("{API} ({name})"), dispatch(make(i))))
        };
        of_type(DeviceType::DiscreteGpu, WgpuDevice::DiscreteGpu)
            .chain(of_type(
                DeviceType::IntegratedGpu,
                WgpuDevice::IntegratedGpu,
            ))
            .collect()
    }

    /// wgpu's default adapter, which may be a software driver, if any exists.
    pub(super) fn default() -> Option<(String, DispatchDevice)> {
        let adapters = adapters();
        let (_, name) = adapters.first()?;
        Some((
            format!("{API} ({name}, default adapter)"),
            dispatch(WgpuDevice::DefaultDevice),
        ))
    }
}

/// Whether a CUDA driver with at least one device is present. The driver
/// library is loaded lazily and its absence is a Rust panic, not a native
/// one, so catching it is sound.
#[cfg(feature = "cuda")]
fn cuda_present() -> bool {
    quietly(cudarc::driver::CudaContext::device_count).is_ok_and(|count| count.is_ok_and(|n| n > 0))
}

fn candidates(choice: BackendChoice) -> Vec<(String, DispatchDevice)> {
    let mut list = Vec::new();
    let auto = choice == BackendChoice::Auto;
    #[cfg(feature = "cuda")]
    if (auto || choice == BackendChoice::Cuda) && cuda_present() {
        list.push((
            "cuda".to_owned(),
            DispatchDevice::Cuda(burn::backend::cuda::CudaDevice::default()),
        ));
    }
    // A missing HIP runtime aborts the process rather than panicking, so it
    // must be detected before a device is created.
    #[cfg(feature = "rocm")]
    if (auto || choice == BackendChoice::Rocm) && cubecl_hip_sys::is_available() {
        list.push((
            "rocm".to_owned(),
            DispatchDevice::Rocm(burn::backend::rocm::RocmDevice::default()),
        ));
    }
    #[cfg(any(feature = "vulkan", feature = "metal", feature = "webgpu"))]
    if auto {
        list.extend(wgpu_adapters::hardware());
    } else if choice == BackendChoice::Wgpu {
        list.extend(wgpu_adapters::default());
    }
    if list.is_empty() && !auto && choice != BackendChoice::Cpu {
        let (name, device) = cpu();
        return vec![(
            format!("{name}; no {choice:?} device found or compiled in"),
            device,
        )];
    }
    list.push(cpu());
    list
}

/// Points cubecl's autotune results and compiled kernels at rverdict's cache,
/// so GPU warm-up is paid once per machine and driver rather than on every
/// start. A host application that configured cubecl first keeps its settings.
#[cfg(any(
    feature = "vulkan",
    feature = "metal",
    feature = "webgpu",
    feature = "cuda",
    feature = "rocm"
))]
fn configure_kernel_cache() {
    use cubecl_runtime::config::cache::CacheConfig;
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};

    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let root = rverdict_core::cache_root().join("kernels");
        let mut config = CubeClRuntimeConfig::default();
        config.autotune.cache = CacheConfig::File(root.clone());
        config.compilation.cache = Some(CacheConfig::File(root));
        let _ = quietly(|| CubeClRuntimeConfig::set(config));
    });
}

/// Selects the first candidate that passes the self-test. CPU always passes.
pub fn select(choice: BackendChoice) -> SelectedBackend {
    #[cfg(any(
        feature = "vulkan",
        feature = "metal",
        feature = "webgpu",
        feature = "cuda",
        feature = "rocm"
    ))]
    configure_kernel_cache();
    let reference = tiny_logits(&DispatchDevice::Flex(FlexDevice), None)
        .expect("the CPU backend runs the tiny model");
    let mut skipped = Vec::new();
    let mut candidates = candidates(choice).into_iter().peekable();
    while let Some((name, device)) = candidates.next() {
        if candidates.peek().is_none() {
            return SelectedBackend {
                device,
                name,
                skipped,
            };
        }
        match self_test(&device, &reference) {
            Ok(()) => {
                return SelectedBackend {
                    device,
                    name,
                    skipped,
                };
            }
            Err(reason) => skipped.push((name, reason)),
        }
    }
    unreachable!("the candidate list always ends with the CPU backend")
}

/// Runs `f`, turning a panic into `Err` without printing it.
fn quietly<T>(f: impl FnOnce() -> T) -> Result<T, Box<dyn std::any::Any + Send>> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(hook);
    outcome
}

/// Runs the tiny model on `device` with the CPU's weights and compares
/// logits. Panics inside the backend (no driver, no adapter) are failures.
fn self_test(device: &DispatchDevice, reference: &(Vec<u8>, Vec<f32>)) -> Result<(), String> {
    let logits = match quietly(|| tiny_logits(device, Some(&reference.0))) {
        Ok(Ok((_, logits))) => logits,
        Ok(Err(e)) => return Err(e),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("backend panicked");
            return Err(message.lines().next().unwrap_or(message).to_owned());
        }
    };
    let worst = logits
        .iter()
        .zip(&reference.1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    if worst.is_finite() && worst < 1e-3 {
        Ok(())
    } else {
        Err(format!("self-test logits differ from CPU by {worst:e}"))
    }
}

/// Tiny-model logits on `device`, initialised randomly or from `weights`.
/// Returns the weights used, so the CPU run can seed the device run.
fn tiny_logits(
    device: &DispatchDevice,
    weights: Option<&[u8]>,
) -> Result<(Vec<u8>, Vec<f32>), String> {
    let config = EncoderConfig::tiny();
    let mut model = config.init_decision_model::<Dispatch>(device);
    let bytes = if let Some(bytes) = weights {
        let mut store = SafetensorsStore::from_bytes(Some(bytes.to_vec()));
        model.load_from(&mut store).map_err(|e| e.to_string())?;
        bytes.to_vec()
    } else {
        let mut store = SafetensorsStore::from_bytes(None);
        model.save_into(&mut store).map_err(|e| e.to_string())?;
        store.get_bytes().map_err(|e| e.to_string())?
    };
    // Long enough to exercise the sliding window, with two option spans.
    let seq = PackedSequence {
        token_ids: vec![1, 10, 11, 12, 13, 14, 15, 16, 17, 2, 4, 20, 21, 4, 22, 2],
        markers: vec![10, 13],
    };
    let input = build_input::<Dispatch>(
        std::slice::from_ref(&seq),
        OptionAttention::Independent,
        config.pad_token_id,
        config.sliding_window,
        device,
    );
    let logits = model
        .option_logits(input, std::slice::from_ref(&seq.markers))
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("{e:?}"))?;
    Ok((bytes, logits))
}
