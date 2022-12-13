struct Mat3
{
   float3 x,y,z;
};

struct Ray
{
   float3 org;
   float3 dir;
   float t;
};

struct Material
{
   float3 col;
   float reflection;
   
};

struct Light
{
   float3  pos;
   float3 col;
   
};

struct RaySphere{
   float3 pos;
   struct Material mat;
   float size;
};

struct RayTriangle{
   float3 vtxA, vtxB, vtxC;
   float3 N;
   float alpha, beta, gamma;
   struct Material mat;
};



inline float3 rt_normalize(float3 v)
{
   float l = sqrt(dot(v,v));
   float3 res;
   if(l==0){
      res = (float3)(1,0,0);
   }
   else{
      res = 1/l*v;
   }
   return res;
}


inline float3 ColorMul(float3 A, float3 B) {
   float3 res; 
   res.x = A.x*B.x; 
   res.y = A.y*B.y; 
   res.z = A.z*B.z; 
   return res;
}

float3 MatVecMul(struct Mat3 rot  , float3 v)  {
   float3 res;
   res.x = dot(rot.x,v);
   res.y = dot(rot.y,v);
   res.z = dot(rot.z,v); 
   return res;
}

inline struct Ray RayInit(float3 pos, float3 dir, float t){
   struct Ray res; 
   res.org=pos; 
   res.dir=dir; 
   res.t=t; 
   return res;
}


bool RaySphereHit(struct RaySphere s, struct Ray* r){
   const float eps = 0.01;
   float3 h = s.pos - r->org;
   float m = dot(h,r->dir);
   float g = m*m - dot(h, h) + s.size*s.size;
   if (g<0) return false;
   float t0 = m - sqrt(g);
   float t1 = m + sqrt(g);
   if (t0>eps && t0<r->t)
   {
      r->t = t0;
      return true;
   }
   else if (t1>eps && t1<r->t)
   {
      r->t = t1;
      return true;
   }
   else{
      return false;
   }
}

float3 RaySphereNormal(struct RaySphere s, float3 p)
{
   return rt_normalize(p - s.pos);
}

float Det(float3 A, float3 B, float3 C){
   float D1 = A.x*(B.y*C.z - C.y*B.z);
   float D2 = B.x*(C.y*A.z - A.y*C.z);
   float D3 = C.x*(A.y*B.z - B.y*A.z);
   
   return  D1 + D2 + D3;   
}


bool RayTriangleHit(struct RayTriangle s, struct Ray* r){
   const float eps = 0.01;
   float3 ab = s.vtxB - s.vtxA;
   float3 ac = s.vtxC - s.vtxA;
   
   float3 aOrg = r->org - s.vtxA;
   float3 pvec = cross(r->dir,ac);
   float D = dot(ab, pvec);
   if(D==0){
      return false;
   }
   float invD = 1/D;
   
   float B = dot(aOrg, pvec)*invD;
   if(B > 1 || B < 0){
      return false;
   }
   float3 qvec = cross(aOrg, ab);
   float C = dot(r->dir, qvec)*invD;
   if(C > 1 || C < 0){
      return false;
   }
   double alpha = 1 -B-C;
   if(alpha>1 || alpha <0){
      return false;
   }

   float t = dot(ac, qvec) * invD;
   if(t>eps  && t<r->t){
      s.alpha = alpha;
      s.beta = B;
      s.gamma = C;
      
      r->t = t;

      return true;
   }

   return false;

}






__kernel void RayTraceSpherePixel(  __global const struct RaySphere* objects, 
                              __global const struct Light* lights,
                              const int obj_size, 
                              const int lights_size, 
                              const int wid,
                              const int hgt,
                              const int maxlev,
                              const float zoom,
                              const struct Mat3 rot,
                              __global unsigned char* pixels)
{
   unsigned int gid = get_global_id(0);
   if(gid>wid*hgt){
      return;
   }
   int i = gid%wid;
   int j = gid/wid;
   float3 org= (float3)(zoom*(i-wid/2) , zoom*(j-hgt/2) , -1000);
   float3 dir = (float3)(0,0,1);
   float3 col = (float3)(0,0,0);
   float coef = 1.0;
   struct Ray ray = RayInit(MatVecMul(rot,org) , MatVecMul(rot,dir), 1e300);
   int k;
   for (int level=0 ; level<maxlev && coef > 0.01 ; level++)
   {
      k = -1;
      for (unsigned int o_id=0 ; o_id<obj_size ; o_id++){
         if(RaySphereHit(objects[o_id], &ray)){
            k=o_id;
         }
      }
      
      if (k<0){
         break;
      }
      
      struct Material mat = objects[k].mat;
      float3 rdir = ray.t * ray.dir;
      float3 P = ray.org + rdir;
      float3 N = RaySphereNormal(objects[k], P);

      col = col+(coef * (float)0.1 * mat.col);
      for (unsigned int l_id=0 ; l_id<lights_size ; l_id++)
      {
         float3 L = rt_normalize(lights[l_id].pos-P);
         if (dot(N,L)<=0) continue;
         struct Ray ray1= RayInit(P , L,1e300);
         bool shadow = false;
         for (unsigned int o_id=0 ; o_id<obj_size && !shadow; o_id++){
            shadow = RaySphereHit(objects[o_id], &ray1);
         }
         if (!shadow) {
            col = col+( dot(L, N) * coef * ColorMul(lights[l_id].col, mat.col));
         }
      }

      coef *= mat.reflection;
      float3 rspec = (float)2 * dot(ray.dir,N) * N;
      ray = RayInit(P , rt_normalize(ray.dir-rspec), 1e300);
      
   }
   unsigned int id= 4*gid;
   pixels[id++] = col.x>1 ? 255 : (unsigned int)(255*col.x);
   pixels[id++] = col.y>1 ? 255 : (unsigned int)(255*col.y);
   pixels[id++] = col.z>1 ? 255 : (unsigned int)(255*col.z);
   pixels[id] = 255;
}



__kernel void RayTraceTrianglePixel(  __global const struct RayTriangle* objects, 
                              __global const struct Light* lights,
                              const int obj_size, 
                              const int lights_size, 
                              const int wid,
                              const int hgt,
                              const int maxlev,
                              const float zoom,
                              const struct Mat3 rot,
                              __global unsigned char* pixels)
{
   unsigned int gid = get_global_id(0);
   if(gid>wid*hgt){
      return;
   }
   int i = gid%wid;
   int j = gid/wid;
   float3 org= (float3)(zoom*(i-wid/2) , zoom*(j-hgt/2) , -1000);
   float3 dir = (float3)(0,0,1);
   float3 col = (float3)(0,0,0);
   float coef = 1.0;
   float factor = 0.1;
   struct Ray ray = RayInit(MatVecMul(rot,org) , MatVecMul(rot,dir), 1e300);
   int k;
   for (int level=0 ; level<maxlev && coef > 0.01 ; level++)
   {
      k = -1;
      for (unsigned int o_id=0 ; o_id<obj_size ; o_id++){
         if(RayTriangleHit(objects[o_id], &ray)){
            k=o_id;
            break;
         }
      }
      
      if (k<0){
         break;
      }
      struct Material mat = objects[k].mat;
      float3 rdir = ray.t * ray.dir;
      float3 P = ray.org+rdir;
      float3 N = objects[k].N;

      col = col + (coef* factor * mat.col);
      
   
      for (unsigned int l_id=0 ; l_id<lights_size ; l_id++)
      {
         float3 L = rt_normalize(lights[l_id].pos-P);
         if (dot(N,L)<=0) continue;
         struct Ray ray= RayInit(P , L,1e300);
         bool shadow = false;
         for (unsigned int o_id=0 ; o_id<obj_size && !shadow; o_id++){
            shadow = RayTriangleHit(objects[o_id], &ray);
         }
         if (!shadow) {
            col = col+( dot(L, N) * coef * ColorMul(lights[l_id].col, mat.col));
         }
      }

      coef *= mat.reflection;
      float INdot = dot(ray.dir,N);
      float3 rspec = (float)2 * INdot * N;
      ray = RayInit(P , rt_normalize(ray.dir-rspec), 1e300);

      
      
   }
   unsigned int id= 4*gid;
   pixels[id++] = col.x>1 ? 255 : (unsigned int)(255*col.x);
   pixels[id++] = col.y>1 ? 255 : (unsigned int)(255*col.y);
   pixels[id++] = col.z>1 ? 255 : (unsigned int)(255*col.z);
   pixels[id] = 255;
}


__kernel void RayTraceScenePixel(  __global const struct RayTriangle* triangles, 
                              __global const struct RaySphere* spheres, 
                              __global const struct Light* lights,
                              const int tri_size, 
                              const int spheres_size,
                              const int lights_size, 
                              const int wid,
                              const int hgt,
                              const int maxlev,
                              const float zoom,
                              const struct Mat3 rot,
                              __global unsigned char* pixels)
{
   unsigned int gid = get_global_id(0);
   if(gid>wid*hgt){
      return;
   }
   int i = gid%wid;
   int j = gid/wid;
   float3 org= (float3)(zoom*(i-wid/2) , zoom*(j-hgt/2) , -1000);
   float3 dir = (float3)(0,0,1);
   float3 col = (float3)(0,0,0);
   float coef = 1.0;
   float factor = 0.1;
   struct Ray ray = RayInit(MatVecMul(rot,org) , MatVecMul(rot,dir), 1e300);
   int k,l;
   for (int level=0 ; level<maxlev && coef > 0.01 ; level++)
   {
      k = -1;
      l = -1;
      for (unsigned int o_id=0 ; o_id<tri_size ; o_id++){
         if(RayTriangleHit(triangles[o_id], &ray)){
            k=o_id;
         
         }
      }
      
      for (unsigned int o_id=0 ; o_id<spheres_size ; o_id++){
         if(RaySphereHit(spheres[o_id], &ray)){
            l=o_id;
         
         }
      
      }
      if (k<0 && l<0){
         break;
      }
      struct Material mat;
      float3 N;
      
      float3 rdir = ray.t * ray.dir;
      float3 P = ray.org+rdir;
      if(l>=0){
         mat = spheres[l].mat;
         N = RaySphereNormal(spheres[l], P);
      }
      else if(k>=0){
         mat = triangles[k].mat;
         N = triangles[k].N;
      }
       
      col = col + (coef* factor * mat.col);
      
   
      for (unsigned int l_id=0 ; l_id<lights_size ; l_id++)
      {
         float3 L = rt_normalize(lights[l_id].pos-P);
         if (dot(N,L)<=0) continue;
         struct Ray ray= RayInit(P , L,1e300);
         bool shadow = false;
         for (unsigned int o_id=0 ; o_id<tri_size && !shadow; o_id++){
            shadow = RayTriangleHit(triangles[o_id], &ray);
         }
         if(!shadow){
            for (unsigned int o_id=0 ; o_id<spheres_size && !shadow; o_id++){
               shadow = RaySphereHit(spheres[o_id], &ray);
            }
         }
         if (!shadow) {
            col = col+( dot(L, N) * coef * ColorMul(lights[l_id].col, mat.col));
         }
      }

      coef *= mat.reflection;
      float INdot = dot(ray.dir,N);
      float3 rspec = (float)2 * INdot * N;
      ray = RayInit(P , rt_normalize(ray.dir-rspec), 1e300);
      
      
   }
   unsigned int id= 4*gid;
   pixels[id++] = col.x>1 ? 255 : (unsigned int)(255*col.x);
   pixels[id++] = col.y>1 ? 255 : (unsigned int)(255*col.y);
   pixels[id++] = col.z>1 ? 255 : (unsigned int)(255*col.z);
   pixels[id] = 255;
}



