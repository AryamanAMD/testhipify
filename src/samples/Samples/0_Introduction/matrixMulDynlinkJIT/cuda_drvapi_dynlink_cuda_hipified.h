/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef __cuda_drvapi_dynlink_cuda_h__
#define __cuda_drvapi_dynlink_cuda_h__

#include <stdlib.h>


#define __cuda_cuda_h__ 1

/**
 * CUDA API versioning support
 */
#define __CUDA_API_VERSION 5000

/**
 * \defgroup CUDA_DRIVER CUDA Driver API
 *
 * This section describes the low-level CUDA driver application programming
 * interface.
 *
 * @{
 */

/**
 * \defgroup CUDA_TYPES Data types used by CUDA driver
 * @{
 */

/**
 * CUDA API version number
 */
#define CUDA_VERSION 3020 /* 3.2 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA device pointer
 */
#if __CUDA_API_VERSION >= 3020

#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long hipDeviceptr_t;
#else
typedef unsigned int hipDeviceptr_t;
#endif

#endif /* __CUDA_API_VERSION >= 3020 */

typedef int hipDevice_t;                                     /**< CUDA device */
typedef struct ihipCtx_t *hipCtx_t;                       /**< CUDA context */
typedef struct ihipModule_t *hipModule_t;                        /**< CUDA module */
typedef struct ihipModuleSymbol_t *hipFunction_t;                     /**< CUDA function */
typedef struct hipArray *hipArray_t;                       /**< CUDA array */
typedef struct hipMipmappedArray *hipMipmappedArray_t;     /**< CUDA mipmapped array */
typedef struct textureReference *hipTexRef;                     /**< CUDA texture reference */
typedef struct CUsurfref_st *CUsurfref;                   /**< CUDA surface reference */
typedef struct ihipEvent_t *hipEvent_t;                       /**< CUDA event */
typedef struct ihipStream_t *hipStream_t;                     /**< CUDA stream */
typedef struct hipGraphicsResource *hipGraphicsResource_t; /**< CUDA graphics interop resource */
typedef unsigned long long hipTextureObject_t;                   /**< CUDA texture object */
typedef unsigned long long hipSurfaceObject_t;                  /**< CUDA surface object */

typedef struct hipUUID_t                                  /**< CUDA definition of UUID */
{
    char bytes[16];
} hipUUID;

/**
 * Context creation flags
 */
typedef enum CUctx_flags_enum
{
    hipDeviceScheduleAuto          = 0x00, /**< Automatic scheduling */
    hipDeviceScheduleSpin          = 0x01, /**< Set spin as default scheduling */
    hipDeviceScheduleYield         = 0x02, /**< Set yield as default scheduling */
    hipDeviceScheduleBlockingSync = 0x04, /**< Set blocking synchronization as default scheduling */
    hipDeviceScheduleBlockingSync       = 0x04, /**< Set blocking synchronization as default scheduling \deprecated */
    hipDeviceMapHost            = 0x08, /**< Support mapped pinned allocations */
    hipDeviceLmemResizeToMax  = 0x10, /**< Keep local memory allocation after launch */
#if __CUDA_API_VERSION < 4000
    hipDeviceScheduleMask          = 0x03,
    CU_CTX_FLAGS_MASK          = 0x1f
#else
    hipDeviceScheduleMask          = 0x07,
    CU_CTX_PRIMARY             = 0x20, /**< Initialize and return the primary context */
    CU_CTX_FLAGS_MASK          = 0x3f
#endif
} CUctx_flags;

/**
 * Event creation flags
 */
typedef enum CUevent_flags_enum
{
    hipEventDefault        = 0, /**< Default event flag */
    hipEventBlockingSync  = 1, /**< Event uses blocking synchronization */
    hipEventDisableTiming = 2  /**< Event will not record timing data */
} CUevent_flags;

/**
 * Array formats
 */
typedef enum hipArray_Format
{
    HIP_AD_FORMAT_UNSIGNED_INT8  = 0x01, /**< Unsigned 8-bit integers */
    HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
    HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
    HIP_AD_FORMAT_SIGNED_INT8    = 0x08, /**< Signed 8-bit integers */
    HIP_AD_FORMAT_SIGNED_INT16   = 0x09, /**< Signed 16-bit integers */
    HIP_AD_FORMAT_SIGNED_INT32   = 0x0a, /**< Signed 32-bit integers */
    HIP_AD_FORMAT_HALF           = 0x10, /**< 16-bit floating point */
    HIP_AD_FORMAT_FLOAT          = 0x20  /**< 32-bit floating point */
} hipArray_Format;

/**
 * Texture reference addressing modes
 */
typedef enum HIPaddress_mode_enum
{
    HIP_TR_ADDRESS_MODE_WRAP   = 0, /**< Wrapping address mode */
    HIP_TR_ADDRESS_MODE_CLAMP  = 1, /**< Clamp to edge address mode */
    HIP_TR_ADDRESS_MODE_MIRROR = 2, /**< Mirror address mode */
    HIP_TR_ADDRESS_MODE_BORDER = 3  /**< Border address mode */
} HIPaddress_mode;

/**
 * Texture reference filtering modes
 */
typedef enum HIPfilter_mode_enum
{
    HIP_TR_FILTER_MODE_POINT  = 0, /**< Point filter mode */
    HIP_TR_FILTER_MODE_LINEAR = 1  /**< Linear filter mode */
} HIPfilter_mode;

/**
 * Device properties
 */
typedef enum hipDeviceAttribute_t
{
    hipDeviceAttributeMaxThreadsPerBlock = 1,              /**< Maximum number of threads per block */
    hipDeviceAttributeMaxBlockDimX = 2,                    /**< Maximum block dimension X */
    hipDeviceAttributeMaxBlockDimY = 3,                    /**< Maximum block dimension Y */
    hipDeviceAttributeMaxBlockDimZ = 4,                    /**< Maximum block dimension Z */
    hipDeviceAttributeMaxGridDimX = 5,                     /**< Maximum grid dimension X */
    hipDeviceAttributeMaxGridDimY = 6,                     /**< Maximum grid dimension Y */
    hipDeviceAttributeMaxGridDimZ = 7,                     /**< Maximum grid dimension Z */
    hipDeviceAttributeMaxSharedMemoryPerBlock = 8,        /**< Maximum shared memory available per block in bytes */
    hipDeviceAttributeMaxSharedMemoryPerBlock = 8,            /**< Deprecated, use hipDeviceAttributeMaxSharedMemoryPerBlock */
    hipDeviceAttributeTotalConstantMemory = 9,              /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    hipDeviceAttributeWarpSize = 10,                         /**< Warp size in threads */
    hipDeviceAttributeMaxPitch = 11,                         /**< Maximum pitch in bytes allowed by memory copies */
    hipDeviceAttributeMaxRegistersPerBlock = 12,           /**< Maximum number of 32-bit registers available per block */
    hipDeviceAttributeMaxRegistersPerBlock = 12,               /**< Deprecated, use hipDeviceAttributeMaxRegistersPerBlock */
    hipDeviceAttributeClockRate = 13,                        /**< Peak clock frequency in kilohertz */
    hipDeviceAttributeTextureAlignment = 14,                 /**< Alignment requirement for textures */
    hipDeviceAttributeAsyncEngineCount = 15,                       /**< Device can possibly copy memory and execute a kernel concurrently */
    hipDeviceAttributeMultiprocessorCount = 16,              /**< Number of multiprocessors on device */
    hipDeviceAttributeKernelExecTimeout = 17,               /**< Specifies whether there is a run time limit on kernels */
    hipDeviceAttributeIntegrated = 18,                        /**< Device is integrated with host memory */
    hipDeviceAttributeCanMapHostMemory = 19,               /**< Device can map host memory into CUDA address space */
    hipDeviceAttributeComputeMode = 20,                      /**< Compute mode (See ::hipComputeMode for details) */
    hipDeviceAttributeMaxTexture1DWidth = 21,           /**< Maximum 1D texture width */
    hipDeviceAttributeMaxTexture2DWidth = 22,           /**< Maximum 2D texture width */
    hipDeviceAttributeMaxTexture2DHeight = 23,          /**< Maximum 2D texture height */
    hipDeviceAttributeMaxTexture3DWidth = 24,           /**< Maximum 3D texture width */
    hipDeviceAttributeMaxTexture3DHeight = 25,          /**< Maximum 3D texture height */
    hipDeviceAttributeMaxTexture3DDepth = 26,           /**< Maximum 3D texture depth */
    hipDeviceAttributeMaxTexture2DLayered = 27,     /**< Maximum texture array width */
    hipDeviceAttributeMaxTexture2DLayered = 28,    /**< Maximum texture array height */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29, /**< Maximum slices in a texture array */
    hipDeviceAttributeSurfaceAlignment = 30,                 /**< Alignment requirement for surfaces */
    hipDeviceAttributeConcurrentKernels = 31,                /**< Device can possibly execute multiple kernels concurrently */
    hipDeviceAttributeEccEnabled = 32,                       /**< Device has ECC support enabled */
    hipDeviceAttributePciBusId = 33,                        /**< PCI bus ID of the device */
    hipDeviceAttributePciDeviceId = 34,                     /**< PCI device ID of the device */
    hipDeviceAttributeTccDriver = 35,                         /**< Device is using TCC driver model */
    hipDeviceAttributeComputeCapabilityMajor = 75,          /**< Major compute capability version number */
    hipDeviceAttributeComputeCapabilityMinor = 76           /**< Minor compute capability version number */
#if __CUDA_API_VERSION >= 4000
                                     , hipDeviceAttributeMemoryClockRate = 36,                 /**< Peak memory clock frequency in kilohertz */
    hipDeviceAttributeMemoryBusWidth = 37,           /**< Global memory bus width in bits */
    hipDeviceAttributeL2CacheSize = 38,                     /**< Size of L2 cache in bytes */
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 39,    /**< Maximum resident threads per multiprocessor */
    hipDeviceAttributeAsyncEngineCount = 40,                /**< Number of asynchronous engines */
    hipDeviceAttributeUnifiedAddressing = 41,                /**< Device uses shares a unified address space with the host */
    hipDeviceAttributeMaxTexture1DLayered = 42,   /**< Maximum 1D layered texture width */
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43   /**< Maximum layers in a 1D layered texture */
#endif
} hipDeviceAttribute_t;

/**
 * Legacy device properties
 */
typedef struct CUdevprop_st
{
    int maxThreadsPerBlock;     /**< Maximum number of threads per block */
    int maxThreadsDim[3];       /**< Maximum size of each dimension of a block */
    int maxGridSize[3];         /**< Maximum size of each dimension of a grid */
    int sharedMemPerBlock;      /**< Shared memory available per block in bytes */
    int totalConstantMemory;    /**< Constant memory available on device in bytes */
    int SIMDWidth;              /**< Warp size in threads */
    int memPitch;               /**< Maximum pitch in bytes allowed by memory copies */
    int regsPerBlock;           /**< 32-bit registers available per block */
    int clockRate;              /**< Clock frequency in kilohertz */
    int textureAlign;           /**< Alignment requirement for textures */
} CUdevprop;

/**
 * Function properties
 */
typedef enum hipFunction_attribute
{
    /**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is currently loaded.
     */
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,

    /**
     * The size in bytes of local memory used by each thread of this function.
     */
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,

    /**
     * The number of registers used by each thread of this function.
     */
    HIP_FUNC_ATTRIBUTE_NUM_REGS = 4,

    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for cubins
     * compiled prior to CUDA 3.0.
     */
    HIP_FUNC_ATTRIBUTE_PTX_VERSION = 5,

    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy cubins that do not have a
     * properly-encoded binary architecture version.
     */
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION = 6,

    HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

/**
 * Function cache configurations
 */
typedef enum hipFuncCache_t
{
    hipFuncCachePreferNone    = 0x00, /**< no preference for shared memory or L1 (default) */
    hipFuncCachePreferShared  = 0x01, /**< prefer larger shared memory and smaller L1 cache */
    hipFuncCachePreferL1      = 0x02  /**< prefer larger L1 cache and smaller shared memory */
} hipFuncCache_t;

/**
 * Shared memory configurations
 */
typedef enum hipSharedMemConfig
{
    hipSharedMemBankSizeDefault    = 0x00, /**< set default shared memory bank size */
    hipSharedMemBankSizeFourByte  = 0x01, /**< set shared memory bank width to four bytes */
    hipSharedMemBankSizeEightByte = 0x02  /**< set shared memory bank width to eight bytes */
} hipSharedMemConfig;

/**
 * Memory types
 */
typedef enum hipMemoryType
{
    hipMemoryTypeHost    = 0x01,    /**< Host memory */
    hipMemoryTypeDevice  = 0x02,    /**< Device memory */
    hipMemoryTypeArray   = 0x03     /**< Array memory */
#if __CUDA_API_VERSION >= 4000
                            , hipMemoryTypeUnified = 0x04     /**< Unified device or host memory */
#endif
} hipMemoryType;

/**
 * Compute Modes
 */
typedef enum hipComputeMode
{
    hipComputeModeDefault           = 0,  /**< Default compute mode (Multiple contexts allowed per device) */
    hipComputeModeProhibited        = 2  /**< Compute-prohibited mode (No contexts can be created on this device at this time) */
#if __CUDA_API_VERSION >= 4000
                                       , hipComputeModeExclusiveProcess = 3  /**< Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time) */
#endif
} hipComputeMode;

/**
 * Online compiler options
 */
typedef enum hipJitOption
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int
     */
    HIPRTC_JIT_MAX_REGISTERS = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization fo the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Option type: unsigned int
     */
    HIPRTC_JIT_THREADS_PER_BLOCK,

    /**
     * Returns a float value in the option of the wall clock time, in
     * milliseconds, spent creating the cubin\n
     * Option type: float
     */
    HIPRTC_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messsages from PTXAS
     * that are informational in nature (the buffer size is specified via
     * option ::HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES) \n
     * Option type: char*
     */
    HIPRTC_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages from PTXAS that
     * reflect errors (the buffer size is specified via option
     * ::HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char*
     */
    HIPRTC_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int
     */
    HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int
     */
    HIPRTC_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed
     */
    HIPRTC_JIT_TARGET_FROM_HIPCONTEXT,

    /**
     * Target is chosen based on supplied ::CUjit_target_enum.\n
     * Option type: unsigned int for enumerated type ::CUjit_target_enum
     */
    HIPRTC_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::CUjit_fallback_enum.\n
     * Option type: unsigned int for enumerated type ::CUjit_fallback_enum
     */
    HIPRTC_JIT_FALLBACK_STRATEGY

} hipJitOption;

/**
 * Online compilation targets
 */
typedef enum CUjit_target_enum
{
    CU_TARGET_COMPUTE_20 = 20,       /**< Compute device class 2.0 */
    CU_TARGET_COMPUTE_21 = 21,       /**< Compute device class 2.1 */
    CU_TARGET_COMPUTE_30 = 30,       /**< Compute device class 3.0 */
    CU_TARGET_COMPUTE_32 = 32,       /**< Compute device class 3.2 */
    CU_TARGET_COMPUTE_35 = 35,       /**< Compute device class 3.5 */
    CU_TARGET_COMPUTE_37 = 37,       /**< Compute device class 3.7 */
    CU_TARGET_COMPUTE_50 = 50,       /**< Compute device class 5.0 */
    CU_TARGET_COMPUTE_52 = 52,       /**< Compute device class 5.2 */
    CU_TARGET_COMPUTE_53 = 53,       /**< Compute device class 5.3 */
    CU_TARGET_COMPUTE_60 = 60,       /**< Compute device class 6.0.*/
    CU_TARGET_COMPUTE_61 = 61,       /**< Compute device class 6.1.*/
    CU_TARGET_COMPUTE_62 = 62,       /**< Compute device class 6.2.*/
    CU_TARGET_COMPUTE_70 = 70        /**< Compute device class 7.0.*/
} CUjit_target;

/**
 * Cubin matching fallback strategies
 */
typedef enum CUjit_fallback_enum
{
    CU_PREFER_PTX = 0,  /**< Prefer to compile ptx */
    CU_PREFER_BINARY    /**< Prefer to fall back to compatible binary code */
} CUjit_fallback;

/**
 * Flags to register a graphics resource
 */
typedef enum hipGraphicsRegisterFlags
{
    hipGraphicsRegisterFlagsNone          = 0x00,
    hipGraphicsRegisterFlagsReadOnly     = 0x01,
    hipGraphicsRegisterFlagsWriteDiscard = 0x02,
    hipGraphicsRegisterFlagsSurfaceLoadStore  = 0x04
} hipGraphicsRegisterFlags;

/**
 * Flags for mapping and unmapping interop resources
 */
typedef enum CUgraphicsMapResourceFlags_enum
{
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
} CUgraphicsMapResourceFlags;

/**
 * Array indices for cube faces
 */
typedef enum CUarray_cubemap_face_enum
{
    CU_CUBEMAP_FACE_POSITIVE_X  = 0x00, /**< Positive X face of cubemap */
    CU_CUBEMAP_FACE_NEGATIVE_X  = 0x01, /**< Negative X face of cubemap */
    CU_CUBEMAP_FACE_POSITIVE_Y  = 0x02, /**< Positive Y face of cubemap */
    CU_CUBEMAP_FACE_NEGATIVE_Y  = 0x03, /**< Negative Y face of cubemap */
    CU_CUBEMAP_FACE_POSITIVE_Z  = 0x04, /**< Positive Z face of cubemap */
    CU_CUBEMAP_FACE_NEGATIVE_Z  = 0x05  /**< Negative Z face of cubemap */
} CUarray_cubemap_face;

/**
 * Limits
 */
typedef enum hipLimit_t
{
    hipLimitStackSize        = 0x00, /**< GPU thread stack size */
    hipLimitPrintfFifoSize  = 0x01, /**< GPU printf FIFO size */
    hipLimitMallocHeapSize  = 0x02  /**< GPU malloc heap size */
} hipLimit_t;

/**
 * Resource types
 */
typedef enum HIPresourcetype_enum
{
    HIP_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    HIP_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    HIP_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} HIPresourcetype;

/**
 * Error codes
 */
typedef enum hipError_t
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::hipEventQuery() and ::hipStreamQuery()).
     */
    hipSuccess                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    hipErrorInvalidValue                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    hipErrorOutOfMemory                  = 2,

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::hipInit() or that initialization has failed.
     */
    hipErrorNotInitialized                = 3,

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    hipErrorDeinitialized                  = 4,

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    hipErrorProfilerDisabled           = 5,
    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
    */
    hipErrorProfilerNotInitialized       = 6,
    /**
     * This indicates profiler has already been started and probably
     * hipProfilerStart() is incorrectly called.
    */
    hipErrorProfilerAlreadyStarted       = 7,
    /**
     * This indicates profiler has already been stopped and probably
     * hipProfilerStop() is incorrectly called.
    */
    hipErrorProfilerAlreadyStopped       = 8,
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    hipErrorNoDevice                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    hipErrorInvalidDevice                 = 101,


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    hipErrorInvalidImage                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::hipCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::hipCtxGetApiVersion() for more details.
     */
    hipErrorInvalidContext                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::hipCtxPushCurrent().
     */
    hipErrorContextAlreadyCurrent        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    hipErrorMapFailed                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    hipErrorUnmapFailed                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    hipErrorArrayIsMapped                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    hipErrorAlreadyMapped                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    hipErrorNoBinaryForGpu              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    hipErrorAlreadyAcquired               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    hipErrorNotMapped                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    hipErrorNotMappedAsArray            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    hipErrorNotMappedAsPointer          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    hipErrorECCNotCorrectable              = 214,

    /**
     * This indicates that the ::hipLimit_t passed to the API call is not
     * supported by the active device.
     */
    hipErrorUnsupportedLimit              = 215,

    /**
     * This indicates that the ::hipCtx_t passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    hipErrorContextAlreadyInUse         = 216,

    /**
     * This indicates that the device kernel source is invalid.
     */
    hipErrorInvalidSource                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    hipErrorFileNotFound                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    hipErrorSharedObjectSymbolNotFound = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    hipErrorSharedObjectInitFailed      = 303,

    /**
     * This indicates that an OS call failed.
     */
    hipErrorOperatingSystem               = 304,


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::hipStream_t and ::hipEvent_t.
     */
    hipErrorInvalidHandle                 = 400,


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    hipErrorNotFound                      = 500,


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::hipSuccess (which indicates completion). Calls that
     * may return this value include ::hipEventQuery() and ::hipStreamQuery().
     */
    hipErrorNotReady                      = 600,


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    hipErrorLaunchFailure                  = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    hipErrorLaunchOutOfResources        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::hipDeviceAttributeKernelExecTimeout for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::hipErrorLaunchFailure). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    hipErrorLaunchTimeOut                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    /**
     * This error indicates that a call to ::hipCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    hipErrorPeerAccessAlreadyEnabled = 704,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register memory from a context which has not had peer access
     * enabled yet via ::hipCtxEnablePeerAccess(), or that
     * ::hipCtxDisablePeerAccess() is trying to disable peer access
     * which has not been enabled yet.
     */
    hipErrorPeerAccessNotEnabled    = 705,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register already-registered memory.
     */
    CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706,

    /**
     * This error indicates that a call to ::cuMemPeerUnregister is trying to
     * unregister memory that has not been registered.
     */
    CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707,

    /**
     * This error indicates that ::hipCtxCreate was called with the flag
     * ::CU_CTX_PRIMARY on a device which already has initialized its
     * primary context.
     */
    hipErrorSetOnActiveProcess         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::hipCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    hipErrorContextIsDestroyed           = 709,

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    hipErrorAssert                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::hipCtxEnablePeerAccess().
     */
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::hipHostRegister()
     * has already been registered.
     */
    hipErrorHostMemoryAlreadyRegistered = 712,

    /**
     * This error indicates that the pointer passed to ::hipHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    hipErrorHostMemoryNotRegistered     = 713,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    hipErrorUnknown                        = 999
} hipError_t;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define CUDA_CB __stdcall
#else
#define CUDA_CB
#endif

/**
 * CUDA stream callback
 * \param hStream The stream the callback was added to, as passed to ::hipStreamAddCallback.  May be NULL.
 * \param status ::hipSuccess or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
typedef void (CUDA_CB *hipStreamCallback_t)(hipStream_t hStream, hipError_t status, void *userData);

#if __CUDA_API_VERSION >= 4000
/**
 * If set, host memory is portable between CUDA contexts.
 * Flag for ::hipHostAlloc()
 */
#define hipHostMallocPortable        0x01

/**
 * If set, host memory is mapped into CUDA address space and
 * ::hipHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::hipHostAlloc()
 */
#define hipHostMallocMapped       0x02

/**
 * If set, host memory is allocated as write-combined - fast to write,
 * faster to DMA, slow to read except via SSE4 streaming load instruction
 * (MOVNTDQA).
 * Flag for ::hipHostAlloc()
 */
#define hipHostMallocWriteCombined   0x04

/**
 * If set, host memory is portable between CUDA contexts.
 * Flag for ::hipHostRegister()
 */
#define hipHostRegisterPortable     0x01

/**
 * If set, host memory is mapped into CUDA address space and
 * ::hipHostGetDevicePointer() may be called on the host pointer.
 * Flag for ::hipHostRegister()
 */
#define hipHostRegisterMapped    0x02

/**
 * If set, peer memory is mapped into CUDA address space and
 * ::cuMemPeerGetDevicePointer() may be called on the host pointer.
 * Flag for ::cuMemPeerRegister()
 */
#define CU_MEMPEERREGISTER_DEVICEMAP    0x02
#endif

#if __CUDA_API_VERSION >= 3020

/**
 * 2D memory copy parameters
 */
typedef struct hip_Memcpy2D
{
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */

    hipMemoryType srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    hipDeviceptr_t srcDevice;      /**< Source device pointer */
    hipArray_t srcArray;           /**< Source array reference */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */

    hipMemoryType dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    hipDeviceptr_t dstDevice;      /**< Destination device pointer */
    hipArray_t dstArray;           /**< Destination array reference */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */

    size_t WidthInBytes;        /**< Width of 2D memory copy in bytes */
    size_t Height;              /**< Height of 2D memory copy */
} hip_Memcpy2D;

/**
 * 3D memory copy parameters
 */
typedef struct HIP_MEMCPY3D
{
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */
    size_t srcZ;                /**< Source Z */
    size_t srcLOD;              /**< Source LOD */
    hipMemoryType srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    hipDeviceptr_t srcDevice;      /**< Source device pointer */
    hipArray_t srcArray;           /**< Source array reference */
    void *reserved0;            /**< Must be NULL */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */
    size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */
    size_t dstZ;                /**< Destination Z */
    size_t dstLOD;              /**< Destination LOD */
    hipMemoryType dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    hipDeviceptr_t dstDevice;      /**< Destination device pointer */
    hipArray_t dstArray;           /**< Destination array reference */
    void *reserved1;            /**< Must be NULL */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
    size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
    size_t Height;              /**< Height of 3D memory copy */
    size_t Depth;               /**< Depth of 3D memory copy */
} HIP_MEMCPY3D;

/**
 * 3D memory cross-context copy parameters
 */
typedef struct CUDA_MEMCPY3D_PEER_st
{
    size_t srcXInBytes;         /**< Source X in bytes */
    size_t srcY;                /**< Source Y */
    size_t srcZ;                /**< Source Z */
    size_t srcLOD;              /**< Source LOD */
    hipMemoryType srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    hipDeviceptr_t srcDevice;      /**< Source device pointer */
    hipArray_t srcArray;           /**< Source array reference */
    hipCtx_t srcContext;       /**< Source context (ignored with srcMemoryType is ::hipMemoryTypeArray) */
    size_t srcPitch;            /**< Source pitch (ignored when src is array) */
    size_t srcHeight;           /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    size_t dstXInBytes;         /**< Destination X in bytes */
    size_t dstY;                /**< Destination Y */
    size_t dstZ;                /**< Destination Z */
    size_t dstLOD;              /**< Destination LOD */
    hipMemoryType dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    hipDeviceptr_t dstDevice;      /**< Destination device pointer */
    hipArray_t dstArray;           /**< Destination array reference */
    hipCtx_t dstContext;       /**< Destination context (ignored with dstMemoryType is ::hipMemoryTypeArray) */
    size_t dstPitch;            /**< Destination pitch (ignored when dst is array) */
    size_t dstHeight;           /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    size_t WidthInBytes;        /**< Width of 3D memory copy in bytes */
    size_t Height;              /**< Height of 3D memory copy */
    size_t Depth;               /**< Depth of 3D memory copy */
} CUDA_MEMCPY3D_PEER;

/**
 * Array descriptor
 */
typedef struct HIP_ARRAY_DESCRIPTOR
{
    size_t Width;             /**< Width of array */
    size_t Height;            /**< Height of array */

    hipArray_Format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
} HIP_ARRAY_DESCRIPTOR;

/**
 * 3D array descriptor
 */
typedef struct HIP_ARRAY3D_DESCRIPTOR
{
    size_t Width;             /**< Width of 3D array */
    size_t Height;            /**< Height of 3D array */
    size_t Depth;             /**< Depth of 3D array */

    hipArray_Format Format;    /**< Array format */
    unsigned int NumChannels; /**< Channels per array element */
    unsigned int Flags;       /**< Flags */
} HIP_ARRAY3D_DESCRIPTOR;

#endif /* __CUDA_API_VERSION >= 3020 */

#if __CUDA_API_VERSION >= 5000
/**
 * CUDA Resource descriptor
 */
typedef struct HIP_RESOURCE_DESC_st
{
    HIPresourcetype resType;                   /**< Resource type */

    union
    {
        struct
        {
            hipArray_t hArray;                   /**< CUDA array */
        } array;
        struct
        {
            hipMipmappedArray_t hMipmappedArray; /**< CUDA mipmapped array */
        } mipmap;
        struct
        {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t sizeInBytes;               /**< Size in bytes */
        } linear;
        struct
        {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;            /**< Array format */
            unsigned int numChannels;         /**< Channels per array element */
            size_t width;                     /**< Width of the array in elements */
            size_t height;                    /**< Height of the array in elements */
            size_t pitchInBytes;              /**< Pitch between two rows in bytes */
        } pitch2D;
        struct
        {
            int reserved[32];
        } __reserved;
    } res;

    unsigned int flags;                       /**< Flags (must be zero) */
} HIP_RESOURCE_DESC;

/**
 * Texture descriptor
 */
typedef struct HIP_TEXTURE_DESC_st
{
    HIPaddress_mode addressMode[3];  /**< Address modes */
    HIPfilter_mode filterMode;       /**< Filter mode */
    unsigned int flags;             /**< Flags */
    unsigned int maxAnisotropy;     /**< Maximum anistropy ratio */
    HIPfilter_mode mipmapFilterMode; /**< Mipmap filter mode */
    float mipmapLevelBias;          /**< Mipmap level bias */
    float minMipmapLevelClamp;      /**< Mipmap minimum level clamp */
    float maxMipmapLevelClamp;      /**< Mipmap maximum level clamp */
    int _reserved[16];
} HIP_TEXTURE_DESC;

/**
 * Resource view format
 */
typedef enum HIPresourceViewFormat_enum
{
    HIP_RES_VIEW_FORMAT_NONE          = 0x00, /**< No resource view format (use underlying resource format) */
    HIP_RES_VIEW_FORMAT_UINT_1X8      = 0x01, /**< 1 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X8      = 0x02, /**< 2 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X8      = 0x03, /**< 4 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X8      = 0x04, /**< 1 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X8      = 0x05, /**< 2 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X8      = 0x06, /**< 4 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X16     = 0x07, /**< 1 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X16     = 0x08, /**< 2 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X16     = 0x09, /**< 4 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X16     = 0x0a, /**< 1 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X16     = 0x0b, /**< 2 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X16     = 0x0c, /**< 4 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X32     = 0x0d, /**< 1 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X32     = 0x0e, /**< 2 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X32     = 0x0f, /**< 4 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X32     = 0x10, /**< 1 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X32     = 0x11, /**< 2 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X32     = 0x12, /**< 4 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_FLOAT_1X16    = 0x13, /**< 1 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X16    = 0x14, /**< 2 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X16    = 0x15, /**< 4 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_1X32    = 0x16, /**< 1 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X32    = 0x17, /**< 2 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X32    = 0x18, /**< 4 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1  = 0x19, /**< Block compressed 1 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2  = 0x1a, /**< Block compressed 2 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3  = 0x1b, /**< Block compressed 3 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4  = 0x1c, /**< Block compressed 4 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC4    = 0x1d, /**< Block compressed 4 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5  = 0x1e, /**< Block compressed 5 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC5    = 0x1f, /**< Block compressed 5 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, /**< Block compressed 6 unsigned half-float */
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H   = 0x21, /**< Block compressed 6 signed half-float */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7  = 0x22  /**< Block compressed 7 */
} HIPresourceViewFormat;

/**
 * Resource view descriptor
 */
typedef struct HIP_RESOURCE_VIEW_DESC_st
{
    HIPresourceViewFormat format;   /**< Resource view format */
    size_t width;                  /**< Width of the resource view */
    size_t height;                 /**< Height of the resource view */
    size_t depth;                  /**< Depth of the resource view */
    unsigned int firstMipmapLevel; /**< First defined mipmap level */
    unsigned int lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int firstLayer;       /**< First layer index */
    unsigned int lastLayer;        /**< Last layer index */
    unsigned int _reserved[16];
} HIP_RESOURCE_VIEW_DESC;

/**
 * GPU Direct v3 tokens
 */
typedef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
{
    unsigned long long p2pToken;
    unsigned int vaSpaceToken;
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS;
#endif



/**
 * If set, the CUDA array is a collection of layers, where each layer is either a 1D
 * or a 2D array and the Depth member of HIP_ARRAY3D_DESCRIPTOR specifies the number
 * of layers, not the depth of a 3D array.
 */
#define hipArrayLayered        0x01

/**
 * Deprecated, use hipArrayLayered
 */
#define CUDA_ARRAY3D_2DARRAY        0x01

/**
 * This flag must be set in order to bind a surface reference
 * to the CUDA array
 */
#define hipArraySurfaceLoadStore   0x02

/**
 * Override the texref format with a format inferred from the array.
 * Flag for ::hipTexRefSetArray()
 */
#define HIP_TRSA_OVERRIDE_FORMAT 0x01

/**
 * Read the texture as integers rather than promoting the values to floats
 * in the range [0,1].
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_READ_AS_INTEGER         0x01

/**
 * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_NORMALIZED_COORDINATES  0x02

/**
 * Perform sRGB->linear conversion during texture read.
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_SRGB  0x10

/**
 * End of array terminator for the \p extra parameter to
 * ::hipModuleLaunchKernel
 */
#define HIP_LAUNCH_PARAM_END            ((void*)0x00)

/**
 * Indicator that the next value in the \p extra parameter to
 * ::hipModuleLaunchKernel will be a pointer to a buffer containing all kernel
 * parameters used for launching kernel \p f.  This buffer needs to
 * honor all alignment/padding requirements of the individual parameters.
 * If ::HIP_LAUNCH_PARAM_BUFFER_SIZE is not also specified in the
 * \p extra array, then ::HIP_LAUNCH_PARAM_BUFFER_POINTER will have no
 * effect.
 */
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)

/**
 * Indicator that the next value in the \p extra parameter to
 * ::hipModuleLaunchKernel will be a pointer to a size_t which contains the
 * size of the buffer specified with ::HIP_LAUNCH_PARAM_BUFFER_POINTER.
 * It is required that ::HIP_LAUNCH_PARAM_BUFFER_POINTER also be specified
 * in the \p extra array if the value associated with
 * ::HIP_LAUNCH_PARAM_BUFFER_SIZE is not zero.
 */
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)

/**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define CU_PARAM_TR_DEFAULT -1

/**
 * CUDA API made obselete at API version 3020
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
#define hipDeviceptr_t                  hipDeviceptr_t
#define hip_Memcpy2D             hip_Memcpy2D
#define hip_Memcpy2D                hip_Memcpy2D
#define HIP_MEMCPY3D             HIP_MEMCPY3D
#define HIP_MEMCPY3D                HIP_MEMCPY3D
#define HIP_ARRAY_DESCRIPTOR     HIP_ARRAY_DESCRIPTOR
#define HIP_ARRAY_DESCRIPTOR        HIP_ARRAY_DESCRIPTOR
#define HIP_ARRAY3D_DESCRIPTOR   CUDA_ARRAY3D_DESCRIPTOR_v1_st
#define HIP_ARRAY3D_DESCRIPTOR      CUDA_ARRAY3D_DESCRIPTOR_v1
#endif /* CUDA_FORCE_LEGACY32_INTERNAL */

#if defined(__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION < 3020

typedef unsigned int hipDeviceptr_t;

typedef struct hip_Memcpy2D
{
    unsigned int srcXInBytes;   /**< Source X in bytes */
    unsigned int srcY;          /**< Source Y */
    hipMemoryType srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    hipDeviceptr_t srcDevice;      /**< Source device pointer */
    hipArray_t srcArray;           /**< Source array reference */
    unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */

    unsigned int dstXInBytes;   /**< Destination X in bytes */
    unsigned int dstY;          /**< Destination Y */
    hipMemoryType dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    hipDeviceptr_t dstDevice;      /**< Destination device pointer */
    hipArray_t dstArray;           /**< Destination array reference */
    unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */

    unsigned int WidthInBytes;  /**< Width of 2D memory copy in bytes */
    unsigned int Height;        /**< Height of 2D memory copy */
} hip_Memcpy2D;

typedef struct HIP_MEMCPY3D
{
    unsigned int srcXInBytes;   /**< Source X in bytes */
    unsigned int srcY;          /**< Source Y */
    unsigned int srcZ;          /**< Source Z */
    unsigned int srcLOD;        /**< Source LOD */
    hipMemoryType srcMemoryType; /**< Source memory type (host, device, array) */
    const void *srcHost;        /**< Source host pointer */
    hipDeviceptr_t srcDevice;      /**< Source device pointer */
    hipArray_t srcArray;           /**< Source array reference */
    void *reserved0;            /**< Must be NULL */
    unsigned int srcPitch;      /**< Source pitch (ignored when src is array) */
    unsigned int srcHeight;     /**< Source height (ignored when src is array; may be 0 if Depth==1) */

    unsigned int dstXInBytes;   /**< Destination X in bytes */
    unsigned int dstY;          /**< Destination Y */
    unsigned int dstZ;          /**< Destination Z */
    unsigned int dstLOD;        /**< Destination LOD */
    hipMemoryType dstMemoryType; /**< Destination memory type (host, device, array) */
    void *dstHost;              /**< Destination host pointer */
    hipDeviceptr_t dstDevice;      /**< Destination device pointer */
    hipArray_t dstArray;           /**< Destination array reference */
    void *reserved1;            /**< Must be NULL */
    unsigned int dstPitch;      /**< Destination pitch (ignored when dst is array) */
    unsigned int dstHeight;     /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */

    unsigned int WidthInBytes;  /**< Width of 3D memory copy in bytes */
    unsigned int Height;        /**< Height of 3D memory copy */
    unsigned int Depth;         /**< Depth of 3D memory copy */
} HIP_MEMCPY3D;

typedef struct HIP_ARRAY_DESCRIPTOR
{
    unsigned int Width;         /**< Width of array */
    unsigned int Height;        /**< Height of array */

    hipArray_Format Format;      /**< Array format */
    unsigned int NumChannels;   /**< Channels per array element */
} HIP_ARRAY_DESCRIPTOR;

typedef struct HIP_ARRAY3D_DESCRIPTOR
{
    unsigned int Width;         /**< Width of 3D array */
    unsigned int Height;        /**< Height of 3D array */
    unsigned int Depth;         /**< Depth of 3D array */

    hipArray_Format Format;      /**< Array format */
    unsigned int NumChannels;   /**< Channels per array element */
    unsigned int Flags;         /**< Flags */
} HIP_ARRAY3D_DESCRIPTOR;

#endif /* (__CUDA_API_VERSION_INTERNAL) || __CUDA_API_VERSION < 3020 */

/*
 * If set, the CUDA array contains an array of 2D slices
 * and the Depth member of HIP_ARRAY3D_DESCRIPTOR specifies
 * the number of slices, not the depth of a 3D array.
 */
#define CUDA_ARRAY3D_2DARRAY        0x01

/**
 * This flag must be set in order to bind a surface reference
 * to the CUDA array
 */
#define hipArraySurfaceLoadStore   0x02

/**
 * Override the texref format with a format inferred from the array.
 * Flag for ::hipTexRefSetArray()
 */
#define HIP_TRSA_OVERRIDE_FORMAT 0x01

/**
 * Read the texture as integers rather than promoting the values to floats
 * in the range [0,1].
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_READ_AS_INTEGER         0x01

/**
 * Use normalized texture coordinates in the range [0,1) instead of [0,dim).
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_NORMALIZED_COORDINATES  0x02

/**
 * Perform sRGB->linear conversion during texture read.
 * Flag for ::hipTexRefSetFlags()
 */
#define HIP_TRSF_SRGB  0x10

/**
 * For texture references loaded into the module, use default texunit from
 * texture reference.
 */
#define CU_PARAM_TR_DEFAULT -1

/** @} */ /* END CUDA_TYPES */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

/**
 * \defgroup CUDA_INITIALIZE Initialization
 *
 * This section describes the initialization functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */

/*********************************
 ** Initialization
 *********************************/
typedef hipError_t  CUDAAPI tcuInit(unsigned int Flags);

/*********************************
 ** Driver Version Query
 *********************************/
typedef hipError_t  CUDAAPI tcuDriverGetVersion(int *driverVersion);

/************************************
 **
 **    Device management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuDeviceGet(hipDevice_t *device, int ordinal);
typedef hipError_t  CUDAAPI tcuDeviceGetCount(int *count);
typedef hipError_t  CUDAAPI tcuDeviceGetName(char *name, int len, hipDevice_t dev);
typedef hipError_t  CUDAAPI tcuDeviceComputeCapability(int *major, int *minor, hipDevice_t dev);
#if __CUDA_API_VERSION >= 3020
typedef hipError_t  CUDAAPI tcuDeviceTotalMem(size_t *bytes, hipDevice_t dev);
#else
typedef hipError_t  CUDAAPI tcuDeviceTotalMem(unsigned int *bytes, hipDevice_t dev);
#endif

typedef hipError_t  CUDAAPI tcuDeviceGetProperties(CUdevprop *prop, hipDevice_t dev);
typedef hipError_t  CUDAAPI tcuDeviceGetAttribute(int *pi, hipDeviceAttribute_t attrib, hipDevice_t dev);
typedef hipError_t  CUDAAPI tcuGetErrorString(hipError_t error, const char **pStr);

/************************************
 **
 **    Context management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuCtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev);
typedef hipError_t  CUDAAPI tcuCtxDestroy(hipCtx_t ctx);
typedef hipError_t  CUDAAPI tcuCtxAttach(hipCtx_t *pctx, unsigned int flags);
typedef hipError_t  CUDAAPI tcuCtxDetach(hipCtx_t ctx);
typedef hipError_t  CUDAAPI tcuCtxPushCurrent(hipCtx_t ctx);
typedef hipError_t  CUDAAPI tcuCtxPopCurrent(hipCtx_t *pctx);

typedef hipError_t  CUDAAPI tcuCtxSetCurrent(hipCtx_t ctx);
typedef hipError_t  CUDAAPI tcuCtxGetCurrent(hipCtx_t *pctx);

typedef hipError_t  CUDAAPI tcuCtxGetDevice(hipDevice_t *device);
typedef hipError_t  CUDAAPI tcuCtxSynchronize(void);


/************************************
 **
 **    Module management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuModuleLoad(hipModule_t *module, const char *fname);
typedef hipError_t  CUDAAPI tcuModuleLoadData(hipModule_t *module, const void *image);
typedef hipError_t  CUDAAPI tcuModuleLoadDataEx(hipModule_t *module, const void *image, unsigned int numOptions, hipJitOption *options, void **optionValues);
typedef hipError_t  CUDAAPI tcuModuleLoadFatBinary(hipModule_t *module, const void *fatCubin);
typedef hipError_t  CUDAAPI tcuModuleUnload(hipModule_t hmod);
typedef hipError_t  CUDAAPI tcuModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name);

#if __CUDA_API_VERSION >= 3020
typedef hipError_t  CUDAAPI tcuModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t hmod, const char *name);
#else
typedef hipError_t  CUDAAPI tcuModuleGetGlobal(hipDeviceptr_t *dptr, unsigned int *bytes, hipModule_t hmod, const char *name);
#endif

typedef hipError_t  CUDAAPI tcuModuleGetTexRef(hipTexRef *pTexRef, hipModule_t hmod, const char *name);
typedef hipError_t  CUDAAPI tcuModuleGetSurfRef(CUsurfref *pSurfRef, hipModule_t hmod, const char *name);

/************************************
 **
 **    Memory management
 **
 ***********************************/
#if __CUDA_API_VERSION >= 3020
typedef hipError_t CUDAAPI tcuMemGetInfo(size_t *free, size_t *total);
typedef hipError_t CUDAAPI tcuMemAlloc(hipDeviceptr_t *dptr, size_t bytesize);
typedef hipError_t CUDAAPI tcuMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr);
typedef hipError_t CUDAAPI tcuMemAllocPitch(hipDeviceptr_t *dptr,
                                          size_t *pPitch,
                                          size_t WidthInBytes,
                                          size_t Height,
                                          // size of biggest r/w to be performed by kernels on this memory
                                          // 4, 8 or 16 bytes
                                          unsigned int ElementSizeBytes
                                         );
#else
typedef hipError_t CUDAAPI tcuMemGetInfo(unsigned int *free, unsigned int *total);
typedef hipError_t CUDAAPI tcuMemAlloc(hipDeviceptr_t *dptr, unsigned int bytesize);
typedef hipError_t CUDAAPI tcuMemGetAddressRange(hipDeviceptr_t *pbase, unsigned int *psize, hipDeviceptr_t dptr);
typedef hipError_t CUDAAPI tcuMemAllocPitch(hipDeviceptr_t *dptr,
                                          unsigned int *pPitch,
                                          unsigned int WidthInBytes,
                                          unsigned int Height,
                                          // size of biggest r/w to be performed by kernels on this memory
                                          // 4, 8 or 16 bytes
                                          unsigned int ElementSizeBytes
                                         );
#endif

typedef hipError_t CUDAAPI tcuMemFree(hipDeviceptr_t dptr);

#if __CUDA_API_VERSION >= 3020
typedef hipError_t CUDAAPI tcuMemAllocHost(void **pp, size_t bytesize);
typedef hipError_t CUDAAPI tcuMemHostGetDevicePointer(hipDeviceptr_t *pdptr, void *p, unsigned int Flags);
#else
typedef hipError_t CUDAAPI tcuMemAllocHost(void **pp, unsigned int bytesize);
#endif

typedef hipError_t CUDAAPI tcuMemFreeHost(void *p);
typedef hipError_t CUDAAPI tcuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);

typedef hipError_t CUDAAPI tcuMemHostGetFlags(unsigned int *pFlags, void *p);

#if __CUDA_API_VERSION >= 4010
/**
 * Interprocess Handles
 */
#define HIP_IPC_HANDLE_SIZE 64

typedef struct hipIpcEventHandle_st
{
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcEventHandle_t;

typedef struct hipIpcMemHandle_st
{
    char reserved[HIP_IPC_HANDLE_SIZE];
} hipIpcMemHandle_t;

typedef enum CUipcMem_flags_enum
{
    hipIpcMemLazyEnablePeerAccess = 0x1 /**< Automatically enable peer access between remote devices as needed */
} CUipcMem_flags;

typedef hipError_t CUDAAPI tcuDeviceGetByPCIBusId(hipDevice_t *dev, char *pciBusId);
typedef hipError_t CUDAAPI tcuDeviceGetPCIBusId(char *pciBusId, int len, hipDevice_t dev);
typedef hipError_t CUDAAPI tcuIpcGetEventHandle(hipIpcEventHandle_t *pHandle, hipEvent_t event);
typedef hipError_t CUDAAPI tcuIpcOpenEventHandle(hipEvent_t *phEvent, hipIpcEventHandle_t handle);
typedef hipError_t CUDAAPI tcuIpcGetMemHandle(hipIpcMemHandle_t *pHandle, hipDeviceptr_t dptr);
typedef hipError_t CUDAAPI tcuIpcOpenMemHandle(hipDeviceptr_t *pdptr, hipIpcMemHandle_t handle, unsigned int Flags);
typedef hipError_t CUDAAPI tcuIpcCloseMemHandle(hipDeviceptr_t dptr);
#endif

typedef hipError_t CUDAAPI tcuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
typedef hipError_t CUDAAPI tcuMemHostUnregister(void *p);;
typedef hipError_t CUDAAPI tcuMemcpy(hipDeviceptr_t dst, hipDeviceptr_t src, size_t ByteCount);
typedef hipError_t CUDAAPI tcuMemcpyPeer(hipDeviceptr_t dstDevice, hipCtx_t dstContext, hipDeviceptr_t srcDevice, hipCtx_t srcContext, size_t ByteCount);

/************************************
 **
 **    Synchronous Memcpy
 **
 ** Intra-device memcpy's done with these functions may execute in parallel with the CPU,
 ** but if host memory is involved, they wait until the copy is done before returning.
 **
 ***********************************/

// 1D functions
#if __CUDA_API_VERSION >= 3020
// system <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount);

// device <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, size_t ByteCount);

// device <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice, size_t ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset, size_t ByteCount);

// system <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyAtoH(void *dstHost, hipArray_t srcArray, size_t srcOffset, size_t ByteCount);

// array <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray, size_t srcOffset, size_t ByteCount);
#else
// system <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, unsigned int ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice, unsigned int ByteCount);

// device <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoD(hipDeviceptr_t dstDevice, hipDeviceptr_t srcDevice, unsigned int ByteCount);

// device <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoA(hipArray_t dstArray, unsigned int dstOffset, hipDeviceptr_t srcDevice, unsigned int ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, unsigned int srcOffset, unsigned int ByteCount);

// system <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoA(hipArray_t dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
typedef hipError_t  CUDAAPI tcuMemcpyAtoH(void *dstHost, hipArray_t srcArray, unsigned int srcOffset, unsigned int ByteCount);

// array <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyAtoA(hipArray_t dstArray, unsigned int dstOffset, hipArray_t srcArray, unsigned int srcOffset, unsigned int ByteCount);
#endif

// 2D memcpy

typedef hipError_t  CUDAAPI tcuMemcpy2D(const hip_Memcpy2D *pCopy);
typedef hipError_t  CUDAAPI tcuMemcpy2DUnaligned(const hip_Memcpy2D *pCopy);

// 3D memcpy

typedef hipError_t  CUDAAPI tcuMemcpy3D(const HIP_MEMCPY3D *pCopy);

/************************************
 **
 **    Asynchronous Memcpy
 **
 ** Any host memory involved must be DMA'able (e.g., allocated with hipMemAllocHost).
 ** memcpy's done with these functions execute in parallel with the CPU and, if
 ** the hardware is available, may execute in parallel with the GPU.
 ** Asynchronous memcpy must be accompanied by appropriate stream synchronization.
 **
 ***********************************/

// 1D functions
#if __CUDA_API_VERSION >= 3020
// system <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoDAsync(hipDeviceptr_t dstDevice,
                                             const void *srcHost, size_t ByteCount, hipStream_t hStream);
typedef hipError_t  CUDAAPI tcuMemcpyDtoHAsync(void *dstHost,
                                             hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream);

// device <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoDAsync(hipDeviceptr_t dstDevice,
                                             hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream);

// system <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset,
                                             const void *srcHost, size_t ByteCount, hipStream_t hStream);
typedef hipError_t  CUDAAPI tcuMemcpyAtoHAsync(void *dstHost, hipArray_t srcArray, size_t srcOffset,
                                             size_t ByteCount, hipStream_t hStream);

#else
// system <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoDAsync(hipDeviceptr_t dstDevice,
                                             const void *srcHost, unsigned int ByteCount, hipStream_t hStream);
typedef hipError_t  CUDAAPI tcuMemcpyDtoHAsync(void *dstHost,
                                             hipDeviceptr_t srcDevice, unsigned int ByteCount, hipStream_t hStream);

// device <-> device memory
typedef hipError_t  CUDAAPI tcuMemcpyDtoDAsync(hipDeviceptr_t dstDevice,
                                             hipDeviceptr_t srcDevice, unsigned int ByteCount, hipStream_t hStream);

// system <-> array memory
typedef hipError_t  CUDAAPI tcuMemcpyHtoAAsync(hipArray_t dstArray, unsigned int dstOffset,
                                             const void *srcHost, unsigned int ByteCount, hipStream_t hStream);
typedef hipError_t  CUDAAPI tcuMemcpyAtoHAsync(void *dstHost, hipArray_t srcArray, unsigned int srcOffset,
                                             unsigned int ByteCount, hipStream_t hStream);
#endif

// 2D memcpy
typedef hipError_t  CUDAAPI tcuMemcpy2DAsync(const hip_Memcpy2D *pCopy, hipStream_t hStream);

// 3D memcpy
typedef hipError_t  CUDAAPI tcuMemcpy3DAsync(const HIP_MEMCPY3D *pCopy, hipStream_t hStream);

/************************************
 **
 **    Memset
 **
 ***********************************/
typedef hipError_t  CUDAAPI tcuMemsetD8(hipDeviceptr_t dstDevice, unsigned char uc, unsigned int N);
typedef hipError_t  CUDAAPI tcuMemsetD16(hipDeviceptr_t dstDevice, unsigned short us, unsigned int N);
typedef hipError_t  CUDAAPI tcuMemsetD32(hipDeviceptr_t dstDevice, unsigned int ui, unsigned int N);

#if __CUDA_API_VERSION >= 3020
typedef hipError_t  CUDAAPI tcuMemsetD2D8(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef hipError_t  CUDAAPI tcuMemsetD2D16(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned short us, size_t Width, size_t Height);
typedef hipError_t  CUDAAPI tcuMemsetD2D32(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned int ui, size_t Width, size_t Height);
#else
typedef hipError_t  CUDAAPI tcuMemsetD2D8(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
typedef hipError_t  CUDAAPI tcuMemsetD2D16(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
typedef hipError_t  CUDAAPI tcuMemsetD2D32(hipDeviceptr_t dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
#endif

/************************************
 **
 **    Function management
 **
 ***********************************/


typedef hipError_t CUDAAPI tcuFuncSetBlockShape(hipFunction_t hfunc, int x, int y, int z);
typedef hipError_t CUDAAPI tcuFuncSetSharedSize(hipFunction_t hfunc, unsigned int bytes);
typedef hipError_t CUDAAPI tcuFuncGetAttribute(int *pi, hipFunction_attribute attrib, hipFunction_t hfunc);
typedef hipError_t CUDAAPI tcuFuncSetCacheConfig(hipFunction_t hfunc, hipFuncCache_t config);
typedef hipError_t CUDAAPI tcuFuncSetSharedMemConfig(hipFunction_t hfunc, hipSharedMemConfig config);

typedef hipError_t CUDAAPI tcuLaunchKernel(hipFunction_t f,
                                         unsigned int gridDimX,  unsigned int gridDimY,  unsigned int gridDimZ,
                                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                         unsigned int sharedMemBytes,
                                         hipStream_t hStream, void **kernelParams, void **extra);

/************************************
 **
 **    Array management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuArrayCreate(hipArray_t *pHandle, const HIP_ARRAY_DESCRIPTOR *pAllocateArray);
typedef hipError_t  CUDAAPI tcuArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR *pArrayDescriptor, hipArray_t hArray);
typedef hipError_t  CUDAAPI tcuArrayDestroy(hipArray_t hArray);

typedef hipError_t  CUDAAPI tcuArray3DCreate(hipArray_t *pHandle, const HIP_ARRAY3D_DESCRIPTOR *pAllocateArray);
typedef hipError_t  CUDAAPI tcuArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR *pArrayDescriptor, hipArray_t hArray);

#if __CUDA_API_VERSION >= 5000
typedef hipError_t CUDAAPI tcuMipmappedArrayCreate(hipMipmappedArray_t *pHandle, const HIP_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);
typedef hipError_t CUDAAPI tcuMipmappedArrayGetLevel(hipArray_t *pLevelArray, hipMipmappedArray_t hMipmappedArray, unsigned int level);
typedef hipError_t CUDAAPI tcuMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
#endif


/************************************
 **
 **    Texture reference management
 **
 ***********************************/
typedef hipError_t  CUDAAPI tcuTexRefCreate(hipTexRef *pTexRef);
typedef hipError_t  CUDAAPI tcuTexRefDestroy(hipTexRef hTexRef);

typedef hipError_t  CUDAAPI tcuTexRefSetArray(hipTexRef hTexRef, hipArray_t hArray, unsigned int Flags);

#if __CUDA_API_VERSION >= 3020
typedef hipError_t  CUDAAPI tcuTexRefSetAddress(size_t *ByteOffset, hipTexRef hTexRef, hipDeviceptr_t dptr, size_t bytes);
typedef hipError_t  CUDAAPI tcuTexRefSetAddress2D(hipTexRef hTexRef, const HIP_ARRAY_DESCRIPTOR *desc, hipDeviceptr_t dptr, size_t Pitch);
#else
typedef hipError_t  CUDAAPI tcuTexRefSetAddress(unsigned int *ByteOffset, hipTexRef hTexRef, hipDeviceptr_t dptr, unsigned int bytes);
typedef hipError_t  CUDAAPI tcuTexRefSetAddress2D(hipTexRef hTexRef, const HIP_ARRAY_DESCRIPTOR *desc, hipDeviceptr_t dptr, unsigned int Pitch);
#endif

typedef hipError_t  CUDAAPI tcuTexRefSetFormat(hipTexRef hTexRef, hipArray_Format fmt, int NumPackedComponents);
typedef hipError_t  CUDAAPI tcuTexRefSetAddressMode(hipTexRef hTexRef, int dim, HIPaddress_mode am);
typedef hipError_t  CUDAAPI tcuTexRefSetFilterMode(hipTexRef hTexRef, HIPfilter_mode fm);
typedef hipError_t  CUDAAPI tcuTexRefSetFlags(hipTexRef hTexRef, unsigned int Flags);

typedef hipError_t  CUDAAPI tcuTexRefGetAddress(hipDeviceptr_t *pdptr, hipTexRef hTexRef);
typedef hipError_t  CUDAAPI tcuTexRefGetArray(hipArray_t *phArray, hipTexRef hTexRef);
typedef hipError_t  CUDAAPI tcuTexRefGetAddressMode(HIPaddress_mode *pam, hipTexRef hTexRef, int dim);
typedef hipError_t  CUDAAPI tcuTexRefGetFilterMode(HIPfilter_mode *pfm, hipTexRef hTexRef);
typedef hipError_t  CUDAAPI tcuTexRefGetFormat(hipArray_Format *pFormat, int *pNumChannels, hipTexRef hTexRef);
typedef hipError_t  CUDAAPI tcuTexRefGetFlags(unsigned int *pFlags, hipTexRef hTexRef);

/************************************
 **
 **    Surface reference management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuSurfRefSetArray(CUsurfref hSurfRef, hipArray_t hArray, unsigned int Flags);
typedef hipError_t  CUDAAPI tcuSurfRefGetArray(hipArray_t *phArray, CUsurfref hSurfRef);

/************************************
 **
 **    Parameter management
 **
 ***********************************/

typedef hipError_t  CUDAAPI tcuParamSetSize(hipFunction_t hfunc, unsigned int numbytes);
typedef hipError_t  CUDAAPI tcuParamSeti(hipFunction_t hfunc, int offset, unsigned int value);
typedef hipError_t  CUDAAPI tcuParamSetf(hipFunction_t hfunc, int offset, float value);
typedef hipError_t  CUDAAPI tcuParamSetv(hipFunction_t hfunc, int offset, void *ptr, unsigned int numbytes);
typedef hipError_t  CUDAAPI tcuParamSetTexRef(hipFunction_t hfunc, int texunit, hipTexRef hTexRef);


/************************************
 **
 **    Launch functions
 **
 ***********************************/

typedef hipError_t CUDAAPI tcuLaunch(hipFunction_t f);
typedef hipError_t CUDAAPI tcuLaunchGrid(hipFunction_t f, int grid_width, int grid_height);
typedef hipError_t CUDAAPI tcuLaunchGridAsync(hipFunction_t f, int grid_width, int grid_height, hipStream_t hStream);

/************************************
 **
 **    Events
 **
 ***********************************/
typedef hipError_t CUDAAPI tcuEventCreate(hipEvent_t *phEvent, unsigned int Flags);
typedef hipError_t CUDAAPI tcuEventRecord(hipEvent_t hEvent, hipStream_t hStream);
typedef hipError_t CUDAAPI tcuEventQuery(hipEvent_t hEvent);
typedef hipError_t CUDAAPI tcuEventSynchronize(hipEvent_t hEvent);
typedef hipError_t CUDAAPI tcuEventDestroy(hipEvent_t hEvent);
typedef hipError_t CUDAAPI tcuEventElapsedTime(float *pMilliseconds, hipEvent_t hStart, hipEvent_t hEnd);

/************************************
 **
 **    Streams
 **
 ***********************************/
typedef hipError_t CUDAAPI tcuStreamCreate(hipStream_t *phStream, unsigned int Flags);
typedef hipError_t CUDAAPI tcuStreamWaitEvent(hipStream_t hStream, hipEvent_t hEvent, unsigned int Flags);
typedef hipError_t CUDAAPI tcuStreamAddCallback(hipStream_t hStream, hipStreamCallback_t callback, void *userData, unsigned int flags);

typedef hipError_t CUDAAPI tcuStreamQuery(hipStream_t hStream);
typedef hipError_t CUDAAPI tcuStreamSynchronize(hipStream_t hStream);
typedef hipError_t CUDAAPI tcuStreamDestroy(hipStream_t hStream);

/************************************
 **
 **    Graphics interop
 **
 ***********************************/
typedef hipError_t CUDAAPI tcuGraphicsUnregisterResource(hipGraphicsResource_t resource);
typedef hipError_t CUDAAPI tcuGraphicsSubResourceGetMappedArray(hipArray_t *pArray, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);

#if __CUDA_API_VERSION >= 3020
typedef hipError_t CUDAAPI tcuGraphicsResourceGetMappedPointer(hipDeviceptr_t *pDevPtr, size_t *pSize, hipGraphicsResource_t resource);
#else
typedef hipError_t CUDAAPI tcuGraphicsResourceGetMappedPointer(hipDeviceptr_t *pDevPtr, unsigned int *pSize, hipGraphicsResource_t resource);
#endif

typedef hipError_t CUDAAPI tcuGraphicsResourceSetMapFlags(hipGraphicsResource_t resource, unsigned int flags);
typedef hipError_t CUDAAPI tcuGraphicsMapResources(unsigned int count, hipGraphicsResource_t *resources, hipStream_t hStream);
typedef hipError_t CUDAAPI tcuGraphicsUnmapResources(unsigned int count, hipGraphicsResource_t *resources, hipStream_t hStream);

/************************************
 **
 **    Export tables
 **
 ***********************************/
typedef hipError_t CUDAAPI tcuGetExportTable(const void **ppExportTable, const hipUUID *pExportTableId);

/************************************
 **
 **    Limits
 **
 ***********************************/

typedef hipError_t CUDAAPI tcuCtxSetLimit(hipLimit_t limit, size_t value);
typedef hipError_t CUDAAPI tcuCtxGetLimit(size_t *pvalue, hipLimit_t limit);
typedef hipError_t CUDAAPI tcuCtxGetCacheConfig(hipFuncCache_t *pconfig);
typedef hipError_t CUDAAPI tcuCtxSetCacheConfig(hipFuncCache_t config);
typedef hipError_t CUDAAPI tcuCtxGetSharedMemConfig(hipSharedMemConfig *pConfig);
typedef hipError_t CUDAAPI tcuCtxSetSharedMemConfig(hipSharedMemConfig config);
typedef hipError_t CUDAAPI tcuCtxGetApiVersion(hipCtx_t ctx, unsigned int *version);

/************************************
 **
 **    Profiler
 **
 ***********************************/
typedef hipError_t CUDAAPI tcuProfilerStop(void);

/************************************
 ************************************/

extern hipError_t CUDAAPI hipInit(unsigned int, int cudaVersion);

extern tcuDriverGetVersion             *hipDriverGetVersion;
extern tcuDeviceGet                    *hipDeviceGet;
extern tcuDeviceGetCount               *hipGetDeviceCount;
extern tcuDeviceGetName                *hipDeviceGetName;
extern tcuDeviceComputeCapability      *hipDeviceComputeCapability;
extern tcuDeviceGetProperties          *cuDeviceGetProperties;
extern tcuDeviceGetAttribute           *hipDeviceGetAttribute;
extern tcuGetErrorString               *cuGetErrorString;
extern tcuCtxDestroy                   *hipCtxDestroy;
extern tcuCtxAttach                    *cuCtxAttach;
extern tcuCtxDetach                    *cuCtxDetach;
extern tcuCtxPushCurrent               *hipCtxPushCurrent;
extern tcuCtxPopCurrent                *hipCtxPopCurrent;

extern tcuCtxSetCurrent                *hipCtxSetCurrent;
extern tcuCtxGetCurrent                *hipCtxGetCurrent;

extern tcuCtxGetDevice                 *hipCtxGetDevice;
extern tcuCtxSynchronize               *hipCtxSynchronize;
extern tcuModuleLoad                   *hipModuleLoad;
extern tcuModuleLoadData               *hipModuleLoadData;
extern tcuModuleLoadDataEx             *hipModuleLoadDataEx;
extern tcuModuleLoadFatBinary          *cuModuleLoadFatBinary;
extern tcuModuleUnload                 *hipModuleUnload;
extern tcuModuleGetFunction            *hipModuleGetFunction;
extern tcuModuleGetTexRef              *hipModuleGetTexRef;
extern tcuModuleGetSurfRef             *cuModuleGetSurfRef;
extern tcuMemFreeHost                  *hipHostFree;
extern tcuMemHostAlloc                 *hipHostAlloc;
extern tcuMemHostGetFlags              *hipHostGetFlags;

extern tcuMemHostRegister              *hipHostRegister;
extern tcuMemHostUnregister            *hipHostUnregister;
extern tcuMemcpy                       *cuMemcpy;
extern tcuMemcpyPeer                   *cuMemcpyPeer;

extern tcuDeviceTotalMem               *hipDeviceTotalMem;
extern tcuCtxCreate                    *hipCtxCreate;
extern tcuModuleGetGlobal              *hipModuleGetGlobal;
extern tcuMemGetInfo                   *hipMemGetInfo;
extern tcuMemAlloc                     *hipMalloc;
extern tcuMemAllocPitch                *hipMemAllocPitch;
extern tcuMemFree                      *hipFree;
extern tcuMemGetAddressRange           *hipMemGetAddressRange;
extern tcuMemAllocHost                 *hipMemAllocHost;
extern tcuMemHostGetDevicePointer      *hipHostGetDevicePointer;
extern tcuFuncSetBlockShape            *cuFuncSetBlockShape;
extern tcuFuncSetSharedSize            *cuFuncSetSharedSize;
extern tcuFuncGetAttribute             *hipFuncGetAttribute;
extern tcuFuncSetCacheConfig           *cuFuncSetCacheConfig;
extern tcuFuncSetSharedMemConfig       *cuFuncSetSharedMemConfig;
extern tcuLaunchKernel                 *hipModuleLaunchKernel;
extern tcuArrayDestroy                 *hipArrayDestroy;
extern tcuTexRefCreate                 *cuTexRefCreate;
extern tcuTexRefDestroy                *cuTexRefDestroy;
extern tcuTexRefSetArray               *hipTexRefSetArray;
extern tcuTexRefSetFormat              *hipTexRefSetFormat;
extern tcuTexRefSetAddressMode         *hipTexRefSetAddressMode;
extern tcuTexRefSetFilterMode          *hipTexRefSetFilterMode;
extern tcuTexRefSetFlags               *hipTexRefSetFlags;
extern tcuTexRefGetArray               *hipTexRefGetArray;
extern tcuTexRefGetAddressMode         *hipTexRefGetAddressMode;
extern tcuTexRefGetFilterMode          *hipTexRefGetFilterMode;
extern tcuTexRefGetFormat              *hipTexRefGetFormat;
extern tcuTexRefGetFlags               *hipTexRefGetFlags;
extern tcuSurfRefSetArray              *cuSurfRefSetArray;
extern tcuSurfRefGetArray              *cuSurfRefGetArray;
extern tcuParamSetSize                 *cuParamSetSize;
extern tcuParamSeti                    *cuParamSeti;
extern tcuParamSetf                    *cuParamSetf;
extern tcuParamSetv                    *cuParamSetv;
extern tcuParamSetTexRef               *cuParamSetTexRef;
extern tcuLaunch                       *cuLaunch;
extern tcuLaunchGrid                   *cuLaunchGrid;
extern tcuLaunchGridAsync              *cuLaunchGridAsync;
extern tcuEventCreate                  *hipEventCreateWithFlags;
extern tcuEventRecord                  *hipEventRecord;
extern tcuEventQuery                   *hipEventQuery;
extern tcuEventSynchronize             *hipEventSynchronize;
extern tcuEventDestroy                 *hipEventDestroy;
extern tcuEventElapsedTime             *hipEventElapsedTime;
extern tcuStreamCreate                 *hipStreamCreateWithFlags;
extern tcuStreamQuery                  *hipStreamQuery;
extern tcuStreamWaitEvent              *hipStreamWaitEvent;
extern tcuStreamAddCallback            *hipStreamAddCallback;
extern tcuStreamSynchronize            *hipStreamSynchronize;
extern tcuStreamDestroy                *hipStreamDestroy;
extern tcuGraphicsUnregisterResource         *hipGraphicsUnregisterResource;
extern tcuGraphicsSubResourceGetMappedArray  *hipGraphicsSubResourceGetMappedArray;
extern tcuGraphicsResourceSetMapFlags        *cuGraphicsResourceSetMapFlags;
extern tcuGraphicsMapResources               *hipGraphicsMapResources;
extern tcuGraphicsUnmapResources             *hipGraphicsUnmapResources;
extern tcuGetExportTable                     *cuGetExportTable;
extern tcuCtxSetLimit                        *hipDeviceSetLimit;
extern tcuCtxGetLimit                        *hipDeviceGetLimit;

// These functions could be using the CUDA 3.2 interface (_v2)
extern tcuMemcpyHtoD                   *hipMemcpyHtoD;
extern tcuMemcpyDtoH                   *hipMemcpyDtoH;
extern tcuMemcpyDtoD                   *hipMemcpyDtoD;
extern tcuMemcpyDtoA                   *cuMemcpyDtoA;
extern tcuMemcpyAtoD                   *cuMemcpyAtoD;
extern tcuMemcpyHtoA                   *hipMemcpyHtoA;
extern tcuMemcpyAtoH                   *hipMemcpyAtoH;
extern tcuMemcpyAtoA                   *cuMemcpyAtoA;
extern tcuMemcpy2D                     *hipMemcpyParam2D;
extern tcuMemcpy2DUnaligned            *hipDrvMemcpy2DUnaligned;
extern tcuMemcpy3D                     *hipDrvMemcpy3D;
extern tcuMemcpyHtoDAsync              *hipMemcpyHtoDAsync;
extern tcuMemcpyDtoHAsync              *hipMemcpyDtoHAsync;
extern tcuMemcpyDtoDAsync              *hipMemcpyDtoDAsync;
extern tcuMemcpyHtoAAsync              *cuMemcpyHtoAAsync;
extern tcuMemcpyAtoHAsync              *cuMemcpyAtoHAsync;
extern tcuMemcpy2DAsync                *hipMemcpyParam2DAsync;
extern tcuMemcpy3DAsync                *hipDrvMemcpy3DAsync;
extern tcuMemsetD8                     *hipMemsetD8;
extern tcuMemsetD16                    *hipMemsetD16;
extern tcuMemsetD32                    *hipMemsetD32;
extern tcuMemsetD2D8                   *cuMemsetD2D8;
extern tcuMemsetD2D16                  *cuMemsetD2D16;
extern tcuMemsetD2D32                  *cuMemsetD2D32;
extern tcuArrayCreate                  *hipArrayCreate;
extern tcuArrayGetDescriptor           *cuArrayGetDescriptor;
extern tcuArray3DCreate                *hipArray3DCreate;
extern tcuArray3DGetDescriptor         *cuArray3DGetDescriptor;
extern tcuTexRefSetAddress             *hipTexRefSetAddress;
extern tcuTexRefSetAddress2D           *hipTexRefSetAddress2D;
extern tcuTexRefGetAddress             *hipTexRefGetAddress;
extern tcuGraphicsResourceGetMappedPointer   *hipGraphicsResourceGetMappedPointer;

extern tcuMipmappedArrayCreate         *hipMipmappedArrayCreate;
extern tcuMipmappedArrayGetLevel       *hipMipmappedArrayGetLevel;
extern tcuMipmappedArrayDestroy        *hipMipmappedArrayDestroy;

extern tcuProfilerStop                    *hipProfilerStop;

#ifdef __cplusplus
}
#endif

//#undef __CUDA_API_VERSION

#endif //__cuda_drvapi_dynlink_cuda_h__
