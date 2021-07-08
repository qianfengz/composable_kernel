/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_CONFIG_H_IN
#define GUARD_CONFIG_H_IN

// "_PACKAGE_" to avoid name contentions: the macros like
// HIP_VERSION_MAJOR are defined in hip_version.h.
// clang-format off
#define HIP_PACKAGE_VERSION_MAJOR 3
#define HIP_PACKAGE_VERSION_MINOR 8
#define HIP_PACKAGE_VERSION_PATCH 20371
// clang-format on

#define HIP_PACKAGE_VERSION_FLAT                                                   \
    ((HIP_PACKAGE_VERSION_MAJOR * 1000ULL + HIP_PACKAGE_VERSION_MINOR) * 1000000 + \
     HIP_PACKAGE_VERSION_PATCH)

#define OLC_DEBUG 1

#define OLC_HIP_COMPILER "/opt/rocm/llvm/bin/clang++"
/* #undef EXTRACTKERNEL_BIN */
#define OLC_OFFLOADBUNDLER_BIN "/opt/rocm/llvm/bin/clang-offload-bundler"

#endif
