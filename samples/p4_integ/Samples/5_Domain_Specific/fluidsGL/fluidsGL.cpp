/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// OpenGL Graphics includes
#include <helper_gl.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// LWCA standard includes
#include <lwda_runtime.h>
#include <lwda_gl_interop.h>

// LWCA FFT Libraries
#include <lwfft.h>

// LWCA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_lwda.h>

#include "defines.h"
#include "fluidsGL_kernels.h"

#define MAX_EPSILON_ERROR 1.0f

const char *sSDKname = "fluidsGL";
// LWCA example code that implements the frequency space version of
// Jos Stam's paper 'Stable Fluids' in 2D. This application uses the
// LWCA FFT library (LWFFT) to perform velocity diffusion and to
// force non-divergence in the velocity field at each time step. It uses
// LWCA-OpenGL interoperability to update the particle field directly
// instead of doing a copy to system memory before drawing. Texture is
// used for automatic bilinear interpolation at the velocity advection step.

void cleanup(void);
void reshape(int x, int y);

// LWFFT plan handle
lwfftHandle planr2c;
lwfftHandle planc2r;
static cData *vxfield = NULL;
static cData *vyfield = NULL;

cData *hvfield = NULL;
cData *dvfield = NULL;
static int wWidth = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// Particle data
GLuint vbo = 0;                                  // OpenGL vertex buffer object
struct lwdaGraphicsResource *lwda_vbo_resource;  // handles OpenGL-LWCA exchange
static cData *particles = NULL;  // particle positions in host memory
static int lastx = 0, lasty = 0;

// Texture pitch
size_t tPitch = 0;  // Now this is compatible with gcc in 64-bit

char *ref_file = NULL;
bool g_bQAAddTestForce = true;
int g_iFrameToCompare = 100;
int g_TotalErrors = 0;

bool g_bExitESC = false;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

void autoTest(char **);

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx,
                          float fy, int r);
extern "C" void advectVelocity(cData *v, float *vx, float *vy, int dx, int pdx,
                               int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt,
                               float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx,
                               int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);

void simulateFluids(void) {
  // simulate fluid
  advectVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM,
                 DT);
  diffuseProject(vxfield, vyfield, CPADW, DIM, DT, VIS);
  updateVelocity(dvfield, (float *)vxfield, (float *)vyfield, DIM, RPADW, DIM);
  advectParticles(vbo, dvfield, DIM, DIM, DT);
}

void display(void) {
  if (!ref_file) {
    sdkStartTimer(&timer);
    simulateFluids();
  }

  // render points from vertex buffer
  glClear(GL_COLOR_BUFFER_BIT);
  glColor4f(0, 1, 0, 0.5f);
  glPointSize(1);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LWLL_FACE);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexPointer(2, GL_FLOAT, 0, NULL);
  glDrawArrays(GL_POINTS, 0, DS);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glDisable(GL_TEXTURE_2D);

  if (ref_file) {
    return;
  }

  // Finish timing before swap buffers to avoid refresh sync
  sdkStopTimer(&timer);
  glutSwapBuffers();

  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "Lwca/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
    glutSetWindowTitle(fps);
    fpsCount = 0;
    fpsLimit = (int)MAX(ifps, 1.f);
    sdkResetTimer(&timer);
  }

  glutPostRedisplay();
}

void autoTest(char **argv) {
  CFrameBufferObject *fbo =
      new CFrameBufferObject(wWidth, wHeight, 4, false, GL_TEXTURE_2D);
  g_CheckRender = new CheckFBO(wWidth, wHeight, 4, fbo);
  g_CheckRender->setPixelFormat(GL_RGBA);
  g_CheckRender->setExecPath(argv[0]);
  g_CheckRender->EnableQAReadback(true);

  fbo->bindRenderPath();

  reshape(wWidth, wHeight);

  for (int count = 0; count < g_iFrameToCompare; count++) {
    simulateFluids();

    // add in a little force so the automated testing is interesting.
    if (ref_file) {
      int x = wWidth / (count + 1);
      int y = wHeight / (count + 1);
      float fx = (x / (float)wWidth);
      float fy = (y / (float)wHeight);
      int nx = (int)(fx * DIM);
      int ny = (int)(fy * DIM);

      int ddx = 35;
      int ddy = 35;
      fx = ddx / (float)wWidth;
      fy = ddy / (float)wHeight;
      int spy = ny - FR;
      int spx = nx - FR;

      addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy,
                FR);
      lastx = x;
      lasty = y;
    }
  }

  display();

  fbo->unbindRenderPath();

  // compare to official reference image, printing PASS or FAIL.
  printf("> (Frame %d) Readback BackBuffer\n", 100);
  g_CheckRender->readback(wWidth, wHeight);
  g_CheckRender->savePPM("fluidsGL.ppm", true, NULL);

  if (!g_CheckRender->PPMvsPPM("fluidsGL.ppm", ref_file, MAX_EPSILON_ERROR,
                               0.25f)) {
    g_TotalErrors++;
  }
}

// very simple von neumann middle-square prng.  can't use rand() in -qatest
// mode because its implementation varies across platforms which makes testing
// for consistency in the important parts of this program diffilwlt.
float myrand(void) {
  static int seed = 72191;
  char sq[22];

  if (ref_file) {
    seed *= seed;
    sprintf(sq, "%010d", seed);
    // pull the middle 5 digits out of sq
    sq[8] = 0;
    seed = atoi(&sq[3]);

    return seed / 99999.f;
  } else {
    return rand() / (float)RAND_MAX;
  }
}

void initParticles(cData *p, int dx, int dy) {
  int i, j;

  for (i = 0; i < dy; i++) {
    for (j = 0; j < dx; j++) {
      p[i * dx + j].x = (j + 0.5f + (myrand() - 0.5f)) / dx;
      p[i * dx + j].y = (i + 0.5f + (myrand() - 0.5f)) / dy;
    }
  }
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
    case 27:
      g_bExitESC = true;
#if defined(__APPLE__) || defined(MACOSX)
      exit(EXIT_SUCCESS);
#else
      glutDestroyWindow(glutGetWindow());
      return;
#endif
      break;

    case 'r':
      memset(hvfield, 0, sizeof(cData) * DS);
      lwdaMemcpy(dvfield, hvfield, sizeof(cData) * DS, lwdaMemcpyHostToDevice);

      initParticles(particles, DIM, DIM);

      lwdaGraphicsUnregisterResource(lwda_vbo_resource);

      getLastLwdaError("lwdaGraphicsUnregisterBuffer failed");

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS, particles,
                   GL_DYNAMIC_DRAW_ARB);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      lwdaGraphicsGLRegisterBuffer(&lwda_vbo_resource, vbo,
                                   lwdaGraphicsMapFlagsNone);

      getLastLwdaError("lwdaGraphicsGLRegisterBuffer failed");
      break;

    default:
      break;
  }
}

void click(int button, int updown, int x, int y) {
  lastx = x;
  lasty = y;
  clicked = !clicked;
}

void motion(int x, int y) {
  // Colwert motion coordinates to domain
  float fx = (lastx / (float)wWidth);
  float fy = (lasty / (float)wHeight);
  int nx = (int)(fx * DIM);
  int ny = (int)(fy * DIM);

  if (clicked && nx < DIM - FR && nx > FR - 1 && ny < DIM - FR && ny > FR - 1) {
    int ddx = x - lastx;
    int ddy = y - lasty;
    fx = ddx / (float)wWidth;
    fy = ddy / (float)wHeight;
    int spy = ny - FR;
    int spx = nx - FR;
    addForces(dvfield, DIM, DIM, spx, spy, FORCE * DT * fx, FORCE * DT * fy,
              FR);
    lastx = x;
    lasty = y;
  }

  glutPostRedisplay();
}

void reshape(int x, int y) {
  wWidth = x;
  wHeight = y;
  glViewport(0, 0, x, y);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 1, 0, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glutPostRedisplay();
}

void cleanup(void) {
  lwdaGraphicsUnregisterResource(lwda_vbo_resource);

  deleteTexture();

  // Free all host and device resources
  free(hvfield);
  free(particles);
  lwdaFree(dvfield);
  lwdaFree(vxfield);
  lwdaFree(vyfield);
  lwfftDestroy(planr2c);
  lwfftDestroy(planc2r);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDeleteBuffers(1, &vbo);

  sdkDeleteTimer(&timer);
}

int initGL(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(wWidth, wHeight);
  glutCreateWindow("Compute Stable Fluids");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(click);
  glutMotionFunc(motion);
  glutReshapeFunc(reshape);

  if (!isGLVersionSupported(1, 5)) {
    fprintf(stderr, "ERROR: Support for OpenGL 1.5 is missing");
    fflush(stderr);
    return false;
  }

  if (!areGLExtensionsSupported("GL_ARB_vertex_buffer_object")) {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  int devID;
  lwdaDeviceProp deviceProps;

#if defined(__linux__)
  char *Xstatus = getelw("DISPLAY");
  if (Xstatus == NULL) {
    printf("Waiving exelwtion as X server is not running\n");
    exit(EXIT_WAIVED);
  }
  setelw("DISPLAY", ":0", 0);
#endif

  printf("%s Starting...\n\n", sSDKname);

  printf(
      "NOTE: The LWCA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  // First initialize OpenGL context, so we can properly set the GL for LWCA.
  // This is necessary in order to achieve optimal performance with OpenGL/LWCA
  // interop.
  if (false == initGL(&argc, argv)) {
    exit(EXIT_SUCCESS);
  }

  // use command-line specified LWCA device, otherwise use device with highest
  // Gflops/s
  devID = findLwdaDevice(argc, (const char **)argv);

  // get number of SMs on this GPU
  checkLwdaErrors(lwdaGetDeviceProperties(&deviceProps, devID));
  printf("LWCA device [%s] has %d Multi-Processors\n", deviceProps.name,
         deviceProps.multiProcessorCount);

  // automated build testing harness
  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
  }

  // Allocate and initialize host data
  GLint bsize;

  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  hvfield = (cData *)malloc(sizeof(cData) * DS);
  memset(hvfield, 0, sizeof(cData) * DS);

  // Allocate and initialize device data
  lwdaMallocPitch((void **)&dvfield, &tPitch, sizeof(cData) * DIM, DIM);

  lwdaMemcpy(dvfield, hvfield, sizeof(cData) * DS, lwdaMemcpyHostToDevice);
  // Temporary complex velocity field data
  lwdaMalloc((void **)&vxfield, sizeof(cData) * PDS);
  lwdaMalloc((void **)&vyfield, sizeof(cData) * PDS);

  setupTexture(DIM, DIM);

  // Create particle array
  particles = (cData *)malloc(sizeof(cData) * DS);
  memset(particles, 0, sizeof(cData) * DS);

  initParticles(particles, DIM, DIM);

  // Create LWFFT transform plan configuration
  checkLwdaErrors(lwfftPlan2d(&planr2c, DIM, DIM, LWFFT_R2C));
  checkLwdaErrors(lwfftPlan2d(&planc2r, DIM, DIM, LWFFT_C2R));

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS, particles,
               GL_DYNAMIC_DRAW_ARB);

  glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

  if (bsize != (sizeof(cData) * DS)) goto EXTERR;

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  checkLwdaErrors(lwdaGraphicsGLRegisterBuffer(&lwda_vbo_resource, vbo,
                                               lwdaGraphicsMapFlagsNone));
  getLastLwdaError("lwdaGraphicsGLRegisterBuffer failed");

  if (ref_file) {
    autoTest(argv);
    cleanup();

    printf("[fluidsGL] - Test Results: %d Failures\n", g_TotalErrors);
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);

  } else {
#if defined(__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif
    glutMainLoop();
  }

  if (!ref_file) {
    exit(EXIT_SUCCESS);
  }

  return 0;

EXTERR:
  printf("Failed to initialize GL extensions.\n");

  exit(EXIT_FAILURE);
}
