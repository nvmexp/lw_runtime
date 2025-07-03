/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#ifndef __no_glut_h__
#define __no_glut_h__

inline void glutSwapBuffers() {}
inline void glutPostRedisplay() {}

inline void glutInit(int *, char **) {}
inline void glutInitDisplayString(const char*) {}
inline void glutInitWindowSize(int, int) {}
inline void glutCreateWindow(const char*) {}
inline void glutSetWindowTitle(const char*) {}

void glutReshapeFunc(void (*)(int, int));
void glutDisplayFunc(void (*display)());
void glutKeyboardFunc(void (*)(unsigned char, int, int));
void glutTimerFunc(unsigned int, void (*)(int), int);
void glutIdleFunc(void (*)(void));

void glutMainLoop();

#endif
