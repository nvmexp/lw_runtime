/* colw1d with a stride option. only does valid colwolutions 
   aclwmulates in y
   y is output , yn is output size
   x is input
   k is kernel, kn is kernel size
   stride is colwolution stride
 */
void signal_(colw1d)(real *y, real *x, real *k, const long yn, const long kn, long stride) {
  long yi, ki;
  for (yi = 0; yi < yn; ++yi) {
    real * xi = x + yi * stride;
    for (ki = 0; ki < kn; ++ki) 
      y[yi] += xi[ki] * k[ki];
  }
}
