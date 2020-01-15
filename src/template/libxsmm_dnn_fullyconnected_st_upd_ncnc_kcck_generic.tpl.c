/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Kunal Banerjee (Intel Corp.)
******************************************************************************/
/* size variables, all const */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int extra_K_tasks = (handle->bk == 64) ? 2 : 1;
const int extra_C_tasks = (handle->bc == 64) ? 2 : 1;
const int work = nBlocksIFm * nBlocksOFm * extra_K_tasks * extra_C_tasks;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int mb1 = 0, ifm1ofm1 = 0, ofm1 = 0, ifm1 = 0, ofm2 = 0, ifm2 = 0;

/* Batch reduce related variables */
unsigned long long  blocks = nBlocksMB;

LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc, bk);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
  ofm1 = ifm1ofm1 / (nBlocksIFm*extra_K_tasks*extra_C_tasks);
  ofm2 = (ifm1ofm1 % (nBlocksIFm*extra_K_tasks*extra_C_tasks))/(nBlocksIFm*extra_C_tasks);
  ifm1 = ((ifm1ofm1 % (nBlocksIFm*extra_K_tasks*extra_C_tasks))%(nBlocksIFm*extra_C_tasks))/extra_C_tasks;
  ifm2 = ((ifm1ofm1 % (nBlocksIFm*extra_K_tasks*extra_C_tasks))%(nBlocksIFm*extra_C_tasks))%extra_C_tasks;
  batchreduce_kernel(&LIBXSMM_VLA_ACCESS(4, doutput,  0, ofm1,  0, ofm2*(handle->bk/extra_K_tasks), nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(4, input, 0, ifm1, 0, ifm2*(handle->bc/extra_C_tasks), nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*(handle->bc/extra_C_tasks), ofm2*(handle->bk/extra_K_tasks), nBlocksIFm, bc, bk), &blocks);
}

libxsmm_barrier_wait(handle->barrier, ltid);

