#ifndef RAYTRACE_H
#define RAYTRACE_H
#include "InitGPUcl.h"


//Only with structs
//

//  3x3 matrix
struct Mat3
{
   cl_float3 x,y,z;
};

inline cl_float3 Vec3ScalarMul(float f, cl_float3 A){
   cl_float3 res; 
   res.x = f*A.x; 
   res.y = f*A.y; 
   res.z = f*A.z; 
   return res;
}

inline cl_float3 Vec3Sub(cl_float3 A, cl_float3 B) {
   cl_float3 res;
   res.x = A.x-B.x; 
   res.y = A.y-B.y; 
   res.z = A.z-B.z; 
   return res;
}

inline float Vec3Dot(cl_float3 A, cl_float3 B){
   float res= (A.x*B.x) + (A.y*B.y) + (A.z*B.z);
   return res;
}

inline cl_float3 normalize(cl_float3 v)
{
   float l = sqrt(Vec3Dot(v,v));
   cl_float3 res;
   if(l==0){
      res = (cl_float3){1,0,0};
   }
   else{
      res = Vec3ScalarMul(1/l,v);
   }
   return res;
}

inline cl_float3 CrossProduct(cl_float3 A, cl_float3 B){
   return (cl_float3){A.y * B.z - A.z * B.y, A.z * B.x - A.x * B.z, A.x * B.y - A.y * B.x};
}

//
// Ray 
//
struct Ray
{
   cl_float3 org;
   cl_float3 dir;
   float t;
};

//
//  Material type
//
struct Material
{
   cl_float3 col;
   float reflection;
   
};

inline struct Material MaterialInit(float r, float g, float b, float f){
   struct Material mat;
   mat.col = (cl_float3){r,g,b};
   mat.reflection = f;
   
   return mat;
}


//
//  Light type
//
struct Light
{
   cl_float3  pos;
   cl_float3 col;
   
};

//Sphere
struct RaySphere{
   cl_float3 pos;
   struct Material mat;
   float size;
};

inline struct RaySphere RaySphereInit(cl_float3 pos, float size, struct Material mat){return RaySphere({pos,mat,size});}

// Triangle
struct RayTriangle{
   cl_float3 vtxA;
   cl_float3 vtxB;
   cl_float3 vtxC;
   cl_float3 N;
   float alpha;
   float beta;
   float gamma;
   struct Material mat;
   
};


inline cl_float3 RayTriangleNormal(cl_float3 A, cl_float3 B, cl_float3 C)
{
   cl_float3 BA = Vec3Sub(B,A);
   cl_float3 CA = Vec3Sub(C,A);
   return normalize(CrossProduct(BA, CA));
}

inline struct RayTriangle RayTriangleInit(cl_float3 A, cl_float3 B, cl_float3 C, struct Material mat){
   return RayTriangle({A,B,C,RayTriangleNormal(A,B,C),0,0,0,mat });
}


void RayTracePixel(int k);
#endif