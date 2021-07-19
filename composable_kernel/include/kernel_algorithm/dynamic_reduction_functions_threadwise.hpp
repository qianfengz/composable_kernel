/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef CK_DYNAMIC_REDUCTION_FUNCTIONS_THREADWISE_HPP
#define CK_DYNAMIC_REDUCTION_FUNCTIONS_THREADWISE_HPP

#include "float_type.hpp"

#include "reduction_common.hpp"
#include "dynamic_reduction_operator.hpp"
#include "dynamic_reduction_functions_binop.hpp"

namespace ck {

template <index_t ThreadBufferLen,
          typename opReduce,
          NanPropagation_t nanPropaOpt>
struct ThreadReduce
{
    using compType = typename opReduce::dataType;
    using binop    = detail::binop_with_nan_check<nanPropaOpt, opReduce, compType>;
    using BufferType = StaticBuffer<AddressSpace::Vgpr, compType, ThreadBufferLen>;
    using IdxBufferType = StaticBuffer<AddressSpace::Vgpr, int, ThreadBufferLen>;

    // This interface does not accumulate on indices
    __device__ static void Reduce(const BufferType &thread_buffer, compType& accuData)
    {
        static_for<0, ThreadBufferLen, 1>{}( [&](auto i) {		
            binop::calculate(accuData, thread_buffer[Number<i>{}]);
        } );
    };

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at first-time reduction
    __device__ static void
    Reduce2(const BufferType &thread_buffer, compType& accuData, int& accuIndex, int indexStart)
    {
        static_for<0, ThreadBufferLen, 1>{}( [&](auto i) {		
            int currIndex    = i + indexStart;
            binop::calculate(accuData, thread_buffer[Number<i>{}], accuIndex, currIndex);
        } );
    };

    // This interface accumulates on both data values and indices and
    // is called by Direct_ThreadWise reduction method at second-time reduction
    __device__ static void Reduce3(const BufferType &thread_buffer,
                                   const IdxBufferType &thread_indices_buffer,
                                   compType& accuData,
                                   int& accuIndex)
    {
        static_for<0, ThreadBufferLen, 1>{}( [&](auto i) {		
            binop::calculate(accuData, thread_buffer[Number<i>{}], accuIndex, thread_indices_buffer[Number<i>{}]);
        } );
    };

    // Set the elements in the per-thread buffer to a specific value
    __device__ static void set_buffer_value(BufferType &thread_buffer, compType value)
    {
        static_for<0, ThreadBufferLen, 1>{}( [&](auto i) {		
            thread_buffer(Number<i>{}) = value;
        } );	
    };

    // Execute unary operation on the per-thread buffer elements
    template <typename unary_op_type>
    __device__ static void operate_on_elements(unary_op_type & unary_op, BufferType &thread_buffer)
    {
        static_for<0, ThreadBufferLen, 1>{}( [&](auto i) {		
            unary_op(thread_buffer(Number<i>{}));
        } ); 	
    };
};

}; // end of namespace ck

#endif
