/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

void (*displayFunc)();

void glutReshapeFunc(void (*)(int, int)) {}
void glutKeyboardFunc(void (*)(unsigned char, int, int)) {}
void glutTimerFunc(unsigned int, void (*)(int), int) {}
void glutIdleFunc(void (*)(void)) {}

void glutDisplayFunc(void (*display)())
{
    displayFunc = display;
}

void glutMainLoop()
{
    while (1)
    {
        displayFunc();
    }
}
