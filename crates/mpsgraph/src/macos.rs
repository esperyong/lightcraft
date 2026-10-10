//! The Objective-C side: a [`Net`] built as one MPSGraph on the system's Metal device, and tiles run through it.
//!
//! Layer by layer (NCHW throughout, as the tiles come):
//! - convolution: `convolution2D` with explicit "same" padding and OIHW weights (ONNX's layout), a skip join as a
//!   `concat` along the channels first, then the bias (broadcast add) and the leaky ReLU;
//! - 2 × 2 stride-2 transposed convolution: a 1 × 1 convolution to `(co, dy, dx)` channels followed by the
//!   depth-to-space below (the blocks don't overlap, so this is the same sum);
//! - depth-to-space, CRD order: reshape `[1, c, 2, 2, h, w]`, transpose to `[1, c, h, 2, w, 2]`, reshape `[1, c, 2h, 2w]`
//!   (no reliance on how a built-in op orders the sub-pixels);
//! - 2 × 2 max pooling with stride 2, no padding.
//!
//! In f16 the input and the constants are cast once in the graph, and the output back to f32.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use lightcraft_denoise::net::{Net, Op, TensorId};
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{MTLCommandQueue, MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphPaddingStyle, MPSGraphPooling2DOpDescriptor, MPSGraphTensor,
    MPSGraphTensorData, MPSGraphTensorNamedDataLayout,
};

use crate::Precision;

/// A built graph and what it runs on.
pub struct Graph {
    graph: Retained<MPSGraph>,
    input: Retained<MPSGraphTensor>,
    output: Retained<MPSGraphTensor>,
    device: Retained<MPSGraphDevice>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    in_shape: Retained<MPSShape>,
}

// SAFETY: none of these classes is main-thread-only (MTLDevice and MTLCommandQueue are documented thread-safe; MPSGraph,
// its tensors and devices are plain objects without thread affinity), and `Retained` retains and releases atomically.
// `MpsRunner` keeps the `Graph` behind a `Mutex`, so it is used by one thread at a time; moving it between threads is
// all this allows.
unsafe impl Send for Graph {}

fn metal_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    // a command-line process may get no default device; then the first one there is
    MTLCreateSystemDefaultDevice().or_else(|| MTLCopyAllDevices().firstObject())
}

/// The name of the Metal device MPSGraph would run on, if there is one.
pub fn device_name() -> Option<String> {
    autoreleasepool(|_| metal_device().map(|d| d.name().to_string()))
}

fn shape(dims: &[usize]) -> Retained<MPSShape> {
    let numbers: Vec<Retained<NSNumber>> = dims.iter().map(|&d| NSNumber::new_usize(d)).collect();
    NSArray::from_retained_slice(&numbers)
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Builds the graph of one network: tensors are added in the network's order.
struct Builder<'a> {
    graph: &'a MPSGraph,
    precision: Precision,
}

impl Builder<'_> {
    /// `t` in the type the graph computes in.
    fn computed(&self, t: Retained<MPSGraphTensor>) -> Retained<MPSGraphTensor> {
        match self.precision {
            Precision::F32 => t,
            // SAFETY: `t` is a tensor of this graph; casting f32 to f16 is defined for every value.
            Precision::F16 => unsafe { self.graph.castTensor_toType_name(&t, MPSDataType::Float16, None) },
        }
    }

    /// A constant of `dims` holding `values` (which must be exactly that many).
    fn constant(&self, values: &[f32], dims: &[usize]) -> Result<Retained<MPSGraphTensor>, String> {
        let count = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or("a constant is too large")?;
        if values.len() != count {
            return Err(format!("a layer has {} numbers where its shape needs {count}", values.len()));
        }
        let data = NSData::with_bytes(&bytes(values));
        // SAFETY: `data` holds exactly `count` native-endian f32 values, which is what a Float32 constant of `dims`
        // reads; MPSGraph copies what it needs from `data`, which lives until this returns.
        let t = unsafe { self.graph.constantWithData_shape_dataType(&data, &shape(dims), MPSDataType::Float32) };
        Ok(self.computed(t))
    }

    /// A stride-1 convolution of `src` (`cin` channels) with "same" padding, then the bias and the activation.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        src: &MPSGraphTensor,
        weight: &[f32],
        bias: &[f32],
        cin: usize,
        cout: usize,
        k: usize,
        leaky: Option<f32>,
    ) -> Result<Retained<MPSGraphTensor>, String> {
        if k == 0 || k.is_multiple_of(2) {
            return Err(format!("a {k} × {k} convolution has no centre"));
        }
        let pad = (k - 1) / 2;
        let w = self.constant(weight, &[cout, cin, k, k])?;
        // SAFETY: plain values; the class method returns nil only for invalid ones, checked below.
        let desc = unsafe {
            MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                1,
                1,
                1,
                1,
                1,
                pad,
                pad,
                pad,
                pad,
                MPSGraphPaddingStyle::Explicit,
                MPSGraphTensorNamedDataLayout::NCHW,
                MPSGraphTensorNamedDataLayout::OIHW,
            )
        }
        .ok_or("MPSGraph refused a convolution")?;
        // SAFETY: `src` is NCHW with `cin` channels (the network was validated so channel counts chain) and `w` is
        // OIHW `[cout, cin, k, k]` in the same type, which is what the descriptor says.
        let y = unsafe { self.graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(src, &w, &desc, None) };
        let b = self.constant(bias, &[1, cout, 1, 1])?;
        // SAFETY: `[1, cout, 1, 1]` broadcasts over the convolution's `[1, cout, h, w]`; both are of this graph.
        let y = unsafe { self.graph.additionWithPrimaryTensor_secondaryTensor_name(&y, &b, None) };
        Ok(match leaky {
            // SAFETY: an element-wise op on a tensor of this graph; any finite slope is valid.
            Some(alpha) => unsafe { self.graph.leakyReLUWithTensor_alpha_name(&y, f64::from(alpha), None) },
            None => y,
        })
    }

    /// Depth-to-space with block 2 in CRD order: channel `4c + 2dy + dx` at `(y, x)` goes to channel `c` at
    /// `(2y + dy, 2x + dx)`. `t` is `[1, 4c, side, side]`.
    fn depth_to_space(&self, t: &MPSGraphTensor, c: usize, side: usize) -> Retained<MPSGraphTensor> {
        let permutation: Vec<Retained<NSNumber>> = [0usize, 1, 4, 2, 5, 3].iter().map(|&d| NSNumber::new_usize(d)).collect();
        // SAFETY: the three shapes hold the same number of elements (`4 c side²`), and the permutation is one of the
        // six dimensions of the tensor it is applied to.
        unsafe {
            let split = self.graph.reshapeTensor_withShape_name(t, &shape(&[1, c, 2, 2, side, side]), None);
            let moved = self.graph.transposeTensor_permutation_name(&split, &NSArray::from_retained_slice(&permutation), None);
            self.graph.reshapeTensor_withShape_name(&moved, &shape(&[1, c, 2 * side, 2 * side]), None)
        }
    }
}

/// Side of tensor `t` of `net` on a tile of `tile` cells.
fn side(net: &Net, tile: usize, t: TensorId) -> Result<usize, String> {
    let s = net.shape(t).ok_or("the network reads a tensor it does not make")?;
    let shifted = if s.level >= 0 {
        u32::try_from(s.level).ok().and_then(|l| tile.checked_shr(l))
    } else {
        u32::try_from(-i64::from(s.level)).ok().and_then(|l| tile.checked_shl(l))
    };
    shifted.filter(|&n| n > 0).ok_or_else(|| "a tensor of the network has no size on this tile".into())
}

/// Build `net` for `tile × tile` tiles. Returns the graph and the device's name.
pub fn build(net: &Net, tile: usize, precision: Precision) -> Result<(Graph, String), String> {
    autoreleasepool(|_| {
        let mtl = metal_device().ok_or("this Mac has no Metal device")?;
        let name = mtl.name().to_string();
        let queue = mtl.newCommandQueue().ok_or("Metal could not make a command queue")?;
        // SAFETY: `mtl` is a live Metal device; MPSGraphDevice keeps its own reference.
        let device = unsafe { MPSGraphDevice::deviceWithMTLDevice(&mtl) };
        // SAFETY: a plain constructor.
        let graph = unsafe { MPSGraph::new() };
        let b = Builder { graph: &graph, precision };
        let in_shape = shape(&[1, net.in_channels, tile, tile]);
        // SAFETY: a placeholder of a fully known shape; it is fed Float32 data of exactly that shape in `run`.
        let input = unsafe { graph.placeholderWithShape_dataType_name(Some(&in_shape), MPSDataType::Float32, None) };
        let mut tensors: Vec<Retained<MPSGraphTensor>> = vec![b.computed(input.clone())];
        let get = |tensors: &[Retained<MPSGraphTensor>], t: TensorId| tensors.get(t).cloned().ok_or("the network reads a tensor it does not make");
        for op in net.ops() {
            let made = match op {
                Op::Conv(c) => {
                    let first = get(&tensors, c.src)?;
                    let src = match c.src2 {
                        Some(s2) => {
                            let second = get(&tensors, s2)?;
                            // SAFETY: two NCHW tensors of this graph at the same resolution (validated), joined along
                            // the channels.
                            unsafe { graph.concatTensors_dimension_name(&NSArray::from_slice(&[&*first, &*second]), 1, None) }
                        }
                        None => first,
                    };
                    b.conv(&src, &c.weight, &c.bias, c.cin, c.cout, c.k, c.leaky)?
                }
                Op::ConvT2(c) => {
                    // a 1 × 1 convolution to channels (co, dy, dx), then depth-to-space
                    let (cin, cout) = (c.cin, c.cout);
                    let mut weight = vec![0f32; 4 * cout * cin];
                    for ci in 0..cin {
                        for co in 0..cout {
                            for q in 0..4 {
                                // ONNX ConvTranspose: [cin][cout][dy][dx]; here [(co, dy, dx)][cin]
                                let v = c.weight.get((ci * cout + co) * 4 + q).copied().ok_or("a transposed convolution lacks weights")?;
                                if let Some(w) = weight.get_mut((co * 4 + q) * cin + ci) {
                                    *w = v;
                                }
                            }
                        }
                    }
                    let bias: Vec<f32> =
                        (0..4 * cout).map(|i| c.bias.get(i / 4).copied().ok_or("a transposed convolution lacks biases")).collect::<Result<_, _>>()?;
                    let src = get(&tensors, c.src)?;
                    let y = b.conv(&src, &weight, &bias, cin, 4 * cout, 1, c.leaky)?;
                    b.depth_to_space(&y, cout, side(net, tile, c.src)?)
                }
                Op::MaxPool2 { src } => {
                    let src = get(&tensors, *src)?;
                    // SAFETY: plain values; nil is checked.
                    let desc = unsafe {
                        MPSGraphPooling2DOpDescriptor::descriptorWithKernelWidth_kernelHeight_strideInX_strideInY_paddingStyle_dataLayout(
                            2,
                            2,
                            2,
                            2,
                            MPSGraphPaddingStyle::TF_VALID,
                            MPSGraphTensorNamedDataLayout::NCHW,
                        )
                    }
                    .ok_or("MPSGraph refused a pooling")?;
                    // SAFETY: an NCHW tensor of this graph whose side is even (the tile fits the network's depth).
                    unsafe { graph.maxPooling2DWithSourceTensor_descriptor_name(&src, &desc, None) }
                }
                Op::DepthToSpace2 { src } => {
                    let channels = net.shape(*src).map(|s| s.channels / 4).filter(|&c| c > 0).ok_or("a depth-to-space has too few channels")?;
                    let t = get(&tensors, *src)?;
                    b.depth_to_space(&t, channels, side(net, tile, *src)?)
                }
            };
            tensors.push(made);
        }
        let last = get(&tensors, net.output())?;
        let output = match precision {
            Precision::F32 => last,
            // SAFETY: a tensor of this graph; f16 to f32 is exact.
            Precision::F16 => unsafe { graph.castTensor_toType_name(&last, MPSDataType::Float32, None) },
        };
        Ok((Graph { graph, input, output, device, queue, in_shape }, name))
    })
}

impl Graph {
    /// Run one tile (`input`: the placeholder's shape, f32) and read the `out_floats` numbers of the output.
    pub fn run(&self, input: &[f32], out_floats: usize) -> Result<Vec<f32>, String> {
        autoreleasepool(|_| {
            let data = NSData::with_bytes(&bytes(input));
            // SAFETY: `data` holds `input.len()` native-endian f32 values and the caller checked that this is the
            // number of elements of `in_shape`; MPSGraphTensorData keeps its own reference to `data`.
            let fed = unsafe {
                MPSGraphTensorData::initWithDevice_data_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    &self.device,
                    &data,
                    &self.in_shape,
                    MPSDataType::Float32,
                )
            };
            let feeds = NSDictionary::from_slices(&[&*self.input], &[&*fed]);
            let targets = NSArray::from_slice(&[&*self.output]);
            // SAFETY: the feeds cover the graph's only placeholder with data of its shape and type, and the target is
            // a tensor of this graph. The call waits until the GPU has finished.
            let results = unsafe { self.graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations(&self.queue, &feeds, &targets, None) };
            let got = results.objectForKey(&self.output).ok_or("MPSGraph gave no result")?;
            // SAFETY: a result of the run above.
            let array = unsafe { got.mpsndarray() };
            // the buffer must hold exactly what `readBytes` writes: check the array's type and size first
            // SAFETY: getters of a live array.
            let (data_type, dims) = unsafe { (array.dataType(), array.numberOfDimensions()) };
            if data_type != MPSDataType::Float32 {
                return Err("MPSGraph's result is not f32".into());
            }
            let mut count = 1usize;
            for d in 0..dims {
                // SAFETY: `d` is below the array's number of dimensions.
                let length = unsafe { array.lengthOfDimension(d) };
                count = count.checked_mul(length).ok_or("MPSGraph's result is too large")?;
            }
            if count != out_floats {
                return Err(format!("MPSGraph gave {count} numbers instead of {out_floats}"));
            }
            let mut out = vec![0f32; out_floats];
            let buffer = NonNull::new(out.as_mut_ptr().cast::<c_void>()).ok_or("no buffer for MPSGraph's result")?;
            // SAFETY: `buffer` points to `out_floats` f32 values, which is exactly the array's f32 element count checked
            // above; a null stride means densely packed, in the array's own order (NCHW, as the output is laid out).
            unsafe { array.readBytes_strideBytes(buffer, std::ptr::null_mut()) };
            Ok(out)
        })
    }
}
