use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BackendName {
    /// burn-flex: CPU, gemm + SIMD.
    #[cfg(feature = "flex")]
    Flex,
    /// burn-ndarray: CPU reference backend.
    #[cfg(feature = "ndarray")]
    Ndarray,
    /// burn-cpu: MLIR-compiled CPU kernels.
    #[cfg(feature = "cpu")]
    Cpu,
    /// burn-wgpu on Vulkan.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// burn-dispatch, runtime-selected Vulkan device.
    #[cfg(all(feature = "dispatch", feature = "vulkan"))]
    DispatchVulkan,
    /// burn-dispatch, runtime-selected Flex device.
    #[cfg(all(feature = "dispatch", feature = "flex"))]
    DispatchFlex,
}

/// Calls `$job::<B>(device, args…)` for the backend named at runtime.
macro_rules! dispatch {
    ($name:expr, $($job:ident)::+($($arg:expr),* $(,)?)) => {
        match $name {
            #[cfg(feature = "flex")]
            $crate::backend::BackendName::Flex => $($job)::+::<burn::backend::Flex>(&Default::default(), $($arg),*),
            #[cfg(feature = "ndarray")]
            $crate::backend::BackendName::Ndarray => $($job)::+::<burn::backend::NdArray>(&Default::default(), $($arg),*),
            #[cfg(feature = "cpu")]
            $crate::backend::BackendName::Cpu => $($job)::+::<burn::backend::Cpu>(&Default::default(), $($arg),*),
            #[cfg(feature = "vulkan")]
            $crate::backend::BackendName::Vulkan => $($job)::+::<burn::backend::Vulkan>(&Default::default(), $($arg),*),
            #[cfg(all(feature = "dispatch", feature = "vulkan"))]
            $crate::backend::BackendName::DispatchVulkan => $($job)::+::<burn::Dispatch>(
                &burn::DispatchDevice::Vulkan(Default::default()), $($arg),*),
            #[cfg(all(feature = "dispatch", feature = "flex"))]
            $crate::backend::BackendName::DispatchFlex => $($job)::+::<burn::Dispatch>(
                &burn::DispatchDevice::Flex(Default::default()), $($arg),*),
        }
    };
}
pub(crate) use dispatch;
