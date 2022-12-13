#ifndef InitCL_h
#define InitCL_h

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

void ErrCheckCL(cl_int err);
int  InitGPU(int verbose,cl_device_id& devid,cl_context& context,cl_command_queue& queue);

#endif
