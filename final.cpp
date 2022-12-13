/*
 *  Ray Trace a Scene with a single light source.
 *
 *  Scene consists of a combination of speheres, cubes, OBJ files.
 *  Computations are accelerated using OpenCL.
 *
 *  Key bindings:
 *  m/M        Change Scene that needs to be rendered
 *  n/N        Increase/decrease maximum number of reflections
 *  N          Change Maximum levels of reflection
 *  A          Move the light source by -10 in X-direction
 *  D          Move the light source by +10 in X-direction
 *  S          Move the light source by -10 in Y-direction
 *  W          Move the light source by +10 in Y-direction
 *  arrows     Change view angle
 *  PgDn/PgUp  Zoom in and out
 *  0          Reset view angle
 *  ESC        Exit
 */

using namespace std;

#include "CSCIx239.h"
#include <vector>

#include <iostream>
#include <fstream>
#include "RayTrace.h"
#include "InitGPUcl.h"
#include "loadobj.cpp"
//  OpenCL globals
cl_device_id     devid;
cl_context       context;
cl_command_queue queue;


//  Global variables
int th=-215;                   //  Azimuth of view angle
int ph=-120;                   //  Elevation of view angle
int mode = 0;               // Diferent Scenes
int maxModes = 1;
int nthread;
float tx = 0;                 //Changes light position
float ty = 0;                 //Changes light position
unsigned char* pixels=NULL; //  Pixel array for entire screen

vector<RaySphere> spheres;

vector<Light> lights;
int wid;
int hgt;                //  Screen dimensions
int maxlev=8;               //  Maximum levels
float zoom=1;               //  Zoom level
struct Mat3 rot;                   //  Rotation matrix
int spheres_size=0;
int obj_size = 0;
int lights_size = 0;

//  Rotation matrix
inline struct Mat3 rotmat( float th,float X,float Y,float Z)
{
   //  Normalize axis
   float l = sqrt(X*X+Y*Y+Z*Z);
   float x = X/l;
   float y = Y/l;
   float z = Z/l;
   //  Calculate sin and cos
   float s = sinf(th*3.1415927/180);
   float c = cosf(th*3.1415927/180);
   float C = 1-c;
   //  Rotation matrix
   struct Mat3 rot;
   rot.x.x = C*x*x+c;     rot.x.y = C*x*y+z*s;   rot.x.z = C*z*x-y*s;
   rot.y.x = C*x*y-z*s;   rot.y.y = C*y*y+c;     rot.y.z = C*y*z+x*s;
   rot.z.x = C*z*x+y*s;   rot.z.y = C*y*z-x*s;   rot.z.z = C*z*z+c;
   return rot;
}

// Materials
Material cyan({(cl_float3){0,1,1},0.5});

Material yellow({(cl_float3){1,1,0},0.5});

Material magenta({(cl_float3){1,0,1},0.5});

Material Rglass({(cl_float3){0.96,0.5,0.5},0.5});

Material Gglass({(cl_float3){0.5,0.96,0.5},0.5});

Material Bglass({(cl_float3){0.5,0.5,0.96},0.5});

//
//  Ray trace Scene
//
void RayTraceScene(){
   //wid=640;
   //hgt=480;
   cl_int  err;
   
   cl_mem objects_host = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(RayTriangle)*objects.size(),&objects.at(0),&err);
   if (err) Fatal("Cannot allocate device memory for objects\n");
   cl_mem spheres_host = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(RaySphere)*spheres.size(),&spheres.at(0),&err);
   if (err) Fatal("Cannot allocate device memory for spheres\n");

   cl_mem lights_host = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(Light)*lights.size(),&lights.at(0),&err);
   if (err) Fatal("Cannot allocate device memory for lights\n");
   cl_mem devpix = clCreateBuffer(context,CL_MEM_WRITE_ONLY,4*hgt*wid,NULL,&err);
   if (err) Fatal("Cannot allocate device memory for pixels\n");
   
   string source;
	ifstream file("RayTrace.cl");
	if (!file){
		printf("\nNo OpenCL file found!\n");
      printf("Exiting...\n");
		//system("PAUSE");
		exit(1);
	}
	
   while (!file.eof()){
		char line[256];
		file.getline(line, 255);
   	source += line;
      //source += "\n";
	}

	const char* kernel_source = source.c_str();
   
   //  Compile kernel
   cl_program prog = clCreateProgramWithSource(context,1,&kernel_source,0,&err);
   //const char* compile_opt = "-Werror";
   if (err) Fatal("Cannot create program\n");
   if (clBuildProgram(prog,0,NULL,"-Werror",NULL,NULL))
   {
      char log[1048576];
      if (clGetProgramBuildInfo(prog,devid,CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL))
         Fatal("Cannot get build log\n");
      else
         Fatal("Cannot build program\n%s\n",log);
   }
   //printf("Created program\n");
   cl_kernel kernel = clCreateKernel(prog,"RayTraceScenePixel",&err);
   ErrCheckCL(err);
   if (err) Fatal("Cannot create kernel\n");
   

   //  Set parameters for kernel function RayTraceScenePixel
   if(clSetKernelArg(kernel,0,sizeof(cl_mem), &objects_host)) Fatal("Cannot set kernel parameters objects\n");
   if(clSetKernelArg(kernel,1,sizeof(cl_mem), &spheres_host)) Fatal("Cannot set kernel parameters spheres\n");
   if(clSetKernelArg(kernel,2,sizeof(cl_mem), &lights_host)) Fatal("Cannot set kernel parameters lights\n");
   if(clSetKernelArg(kernel,3,sizeof(int), &obj_size)) Fatal("Cannot set kernel parameters obj_size\n");
   if(clSetKernelArg(kernel,4,sizeof(int), &spheres_size)) Fatal("Cannot set kernel parameters spheres_size\n");
   if(clSetKernelArg(kernel,5,sizeof(int), &lights_size)) Fatal("Cannot set kernel parameters lights_size\n");
   if(clSetKernelArg(kernel,6,sizeof(int), &wid)) Fatal("Cannot set kernel parameters wid\n");
   if(clSetKernelArg(kernel,7,sizeof(int), &hgt)) Fatal("Cannot set kernel parameters hgt\n");
   if(clSetKernelArg(kernel,8,sizeof(int), &maxlev)) Fatal("Cannot set kernel parameters max_lev\n");
   if(clSetKernelArg(kernel,9,sizeof(float), &zoom)) Fatal("Cannot set kernel parameters zoom\n");
   if(clSetKernelArg(kernel,10,sizeof(Mat3), &rot)) Fatal("Cannot set kernel parameters rot\n");
   if (clSetKernelArg(kernel,11,sizeof(cl_mem),&devpix)) Fatal("Cannot set kernel parameter devpix\n");
      
   //  Run kernel
   //  Ray trace scene
   int nblock = wid*hgt/nthread;
   if (nblock*nthread<wid*hgt) nblock++;
   
   size_t Global[1] = {(unsigned int)nblock*nthread};
   size_t Local[1] = {(unsigned int)nthread};
   if (clEnqueueNDRangeKernel(queue,kernel,1,NULL,Global,Local,0,NULL,NULL)) Fatal("Cannot run kernel\n");

   
   //  Release kernel and program
   if (clReleaseKernel(kernel)) Fatal("Cannot release kernel\n");
   if (clReleaseProgram(prog)) Fatal("Cannot release program\n");

   //  Copy pixels from device to host
   if (clEnqueueReadBuffer(queue,devpix,CL_TRUE,0,4*hgt*wid,pixels,0,NULL,NULL)) Fatal("Cannot copy pixels from device to host\n");


   //  Free device memory
   clReleaseMemObject(devpix);


   
}




//
//  Load Obj file as a vector of Triangles
//
void createObject(const char* file, Material mat,
                  int Tx, int Ty, int Tz,
                  int Rx, int Ry, int Rz,
                  int Sx, int Sy, int Sz){
   float M[16];
   mat4identity(M);
   mat4translate(M, Tx, Ty, Tz);
   mat4rotate(M, Rx, 1, 0, 0);
   mat4rotate(M, Ry, 0, 1, 0);
   mat4rotate(M, Rz, 0, 0, 1);
   mat4scale(M, Sx, Sy, Sz);
   LoadOBJFile(file, M, mat);
   return;
}

//
//  Refresh display
//
void display(GLFWwindow* window)
{
   
   Elapsed();
   objects.clear();
   spheres.clear();
   lights.clear();
   
   if(mode==0){
      printf("Mode %d: Scene with two spheres\n",mode);
      spheres.push_back(RaySphereInit((cl_float3){80, 120, -40}, 40,magenta));
      spheres.push_back(RaySphereInit((cl_float3){80, 0, -40}, 80,cyan));
      spheres_size = spheres.size();
      createObject("cube.obj",cyan, 0,0,0,0,0,0,1,1,1);
      obj_size = 0;
      lights.push_back(Light({(cl_float3){0+tx,320+ty,-1000} ,(cl_float3) {1.0,1.0,1.0}}));
      lights_size = lights.size();
      //maxlev = 8;
   }
   else if(mode==1){
      printf("Mode %d: Scene with a sphere and a cube\n",mode);
      spheres.push_back(RaySphereInit((cl_float3){80, 40, -40}, 80,cyan));
      spheres_size = spheres.size();
      createObject("cube.obj",yellow, 80,-150,-40,0,0,0,80,80,80);
      obj_size = objects.size();
      lights.push_back(Light({(cl_float3){0+tx,320+ty,-1000} ,(cl_float3) {1.0,1.0,1.0}}));
      lights_size = lights.size();
      //maxlev = 8;
   }
   else if(mode==2){
      printf("Mode %d: Scene with two cubes \n",mode);
      spheres.push_back(RaySphereInit((cl_float3){80, 40, -40}, 80,cyan));
      spheres_size = 0;
      createObject("cube.obj",Rglass, 80,0,-40,0,0,0,40,40,40);
      
      createObject("cube.obj",Gglass, -100,40,-40,0,0,0,80,80,80);
      obj_size = objects.size();
      lights.push_back(Light({(cl_float3){0+tx,320+ty,-1000} ,(cl_float3) {1.0,1.0,1.0}}));
      lights_size = lights.size();
      //maxlev = 8;
   }
   else if(mode==3){
      printf("Mode %d: Scene with a sphere, cube and coil (OBJ file) \n",mode);
      spheres.push_back(RaySphereInit((cl_float3){80, 40, -40}, 80,cyan));
      spheres_size = spheres.size();
      createObject("coil.obj",Rglass, -100,-230,-40,0,0,0,20,20,20);
      
      createObject("cube.obj",Gglass, -100,40,-40,0,0,0,40,40,40);
      obj_size = objects.size();
      lights.push_back(Light({(cl_float3){0+tx,320+ty,-1000} ,(cl_float3) {1.0,1.0,1.0}}));
      lights_size = lights.size();
      
   }
   else if(mode==4){
      printf("Mode %d: How many Spheres in the scene? \n",mode);
      spheres.push_back(RaySphereInit((cl_float3){0, 0, 0}, 20,cyan));
      spheres.push_back(RaySphereInit((cl_float3){30, 0, -30}, 20,magenta));
      spheres.push_back(RaySphereInit((cl_float3){-40, -30, 100}, 20,Bglass));
      spheres_size = spheres.size();
      //spheres_size = 0;
      
      createObject("box1.obj",Rglass, 0,0,0,0,0,0,150,150,150);
      createObject("box2.obj",Gglass, 0,0,0,0,0,0,150,150,150);
      createObject("box3.obj",yellow, 0,0,0,0,0,0,150,150,150);
      obj_size = objects.size();
      lights.push_back(Light({(cl_float3){650+tx,-390+ty,-1000} ,(cl_float3) {1.0,1.0,1.0}}));
      lights_size = lights.size();
      
   }
   maxModes = 5;
   //  Ray trace scene
   RayTraceScene();
   
   //  Time ray tracing
   float t = Elapsed();
   //  Blit scene to screen
   glWindowPos2i(0,0);
   glDrawPixels(wid,hgt,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
   //  Display
   glWindowPos2i(5,5);
   Print("Size %dx%d Time %.3fs Angle %d,%d Levels %d Light %d,%d,%d",wid,hgt,t,th,ph,maxlev,(int)lights[0].pos.x,(int)lights[0].pos.y,(int)lights[0].pos.z);
   //  Flush
   glFlush();
   glfwSwapBuffers(window);
}



//
//  Set rotation matrix
//
void SetRot(void)
{
   float M[16];
   mat4identity(M);
   mat4rotate(M , ph,1,0,0);
   mat4rotate(M , th,0,1,0);
   //  Copy matrix to row vectors
   rot.x.x = M[0]; rot.x.y = M[4]; rot.x.z = M[8];
   rot.y.x = M[1]; rot.y.y = M[5]; rot.y.z = M[9];
   rot.z.x = M[2]; rot.z.y = M[6]; rot.z.z = M[10];
}

//
//  Key pressed callback
//
void key(GLFWwindow* window,int key,int scancode,int action,int mods)
{
   //  Discard key releases (keeps PRESS and REPEAT)
   if (action==GLFW_RELEASE) return;

   //  Check for shift
   int shift = (mods & GLFW_MOD_SHIFT);

   //  Exit on ESC
   if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window,1);
   //  Reset view angle
   else if (key == GLFW_KEY_0)
      th = ph = 0;
   //  Change level
   else if (key == GLFW_KEY_N)
      maxlev += shift ? -1 : +1;
   // Move the light source 0 in X-direction
   else if (key == GLFW_KEY_A)
      tx -= 10;
   // Move the light source 0 in X-direction
   else if (key == GLFW_KEY_D)
      tx += 10;
   // Move the light source 0 in Y-direction
   else if (key == GLFW_KEY_S)
      ty -= 10;
   // Move the light source 0 in Y-direction
   else if (key == GLFW_KEY_W)
      ty += 10;
   //  Right arrow key - increase angle by 5 degrees
   else if (key == GLFW_KEY_RIGHT)
      th += 5;
   //  Left arrow key - decrease angle by 5 degrees
   else if (key == GLFW_KEY_LEFT)
      th -= 5;
   //  Up arrow key - increase elevation by 5 degrees
   else if (key == GLFW_KEY_UP)
      ph += 5;
   //  Down arrow key - decrease elevation by 5 degrees
   else if (key == GLFW_KEY_DOWN)
      ph -= 5;
   //  Page Up key - increase zoom
   else if (key == GLFW_KEY_PAGE_UP)
      zoom *= 0.9;
   //  Page Down key - decrease zoom
   else if (key == GLFW_KEY_PAGE_DOWN)
      zoom *= 1.1;
   // Change the Scene
   else if(key == GLFW_KEY_M)
   {
      mode = (mode+1)%maxModes;
      if(mode==4){
         maxlev = 1;
         th =-215;
         ph = -110;
      }
   }
   if (maxlev<1) maxlev = 1;
   //  Keep angles to +/-360 degrees
   th %= 360;
   ph %= 360;
   SetRot();
   
   
}

//
//  Window resized callback
//
void reshape(GLFWwindow* window,int width,int height)
{
   //  Get framebuffer dimensions (makes Apple work right)
   glfwGetFramebufferSize(window,&wid,&hgt);
   //  Allocate pixels size of window
   delete pixels;
   pixels = new unsigned char [4*wid*hgt];
   //  Set the viewport to the entire window
   glViewport(0,0, wid,hgt);
}


//
//  Main program with GLFW event loop
//
int main(int argc,char* argv[])
{
   //  Initialize GLFW
   GLFWwindow* window = InitWindow("final: Ajay Narasimha Mopidevi",0,640,480,&reshape,&key);
   int verbose = 1;
   nthread = InitGPU(verbose,devid,context,queue);
   //printf("Threads: %d\n",nthread);
   
   //  Initialize rotation matrix
   SetRot();

   //  Event loop
   ErrCheck("init");
   while(!glfwWindowShouldClose(window))
   {
      //  Display
      display(window);
      //  Wait for events
      glfwWaitEvents();
   }
   //  Shut down GLFW
   glfwDestroyWindow(window);
   glfwTerminate();
   
   
   return 0;
}
