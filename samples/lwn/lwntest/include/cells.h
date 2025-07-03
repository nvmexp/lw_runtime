/*
 * Copyright (c) 2006 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CELLS_H__
#define __CELLS_H__

#include "ogtest.h"

/***************************************************************************************
 * A mechanism leads to mouse based interactivity
 * rows and columns start counting at 0
 ***************************************************************************************/

#define MAX_CELL_ROW_COL 256

void cellInit(void);
int cellAllowed(int col, int row);
int cellAllowedRange(int col1, int row1, int col2, int row2);
int cellInitForTest(void);
void cellInteractionInit(unsigned char value);
#if defined(__cplusplus)
void cellTestInit(unsigned int cols, unsigned int rows, int marginLeft = 0, int marginBottom = 0,
                  int marginRight = 0, int marginTop = 0);
#else
void cellTestInit(unsigned int cols, unsigned int rows, int marginLeft, int marginBottom,
                   int marginRight, int marginTop);
#endif
void cellTestFini(void);

void cellSelectRange(int col1, int row1, int col2, int row2,
                     unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL]);
void cellToggleRange(int col1, int row1, int col2, int row2,
                     unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL]);
void cellSelectBlock(int col1, int row1, int col2, int row2,
                     unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL]);
void cellToggleBlock(int col1, int row1, int col2, int row2,
                     unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL]);

void cellPrintCmdLine(void);

/**
* Reset the cell viewport and scissor
*/
void cellReset(void);

typedef struct { // keep these types in sync with MODS
    int col;
    int row;
} Cell;

typedef struct {
    unsigned int cols;
    unsigned int rows;
    int marginLeft;
    int marginBottom;
    int marginRight;
    int marginTop;
    unsigned char defined; // whether this structure means anything
} CellGridLayout;

extern int cellSelection; // if the cmdline specifically asks for certain cells
extern int cellInteractiveSelection;
extern unsigned char defaultCellGrid[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL];
extern unsigned char interactiveCellGrid[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL];
extern CellGridLayout interactiveGridLayout;

// We check previousGridLayout against interactiveGridLayout to see whether it makse sense for us to retain the grid selection
extern CellGridLayout previousGridLayout;

Cell cellGet(int x, int y, const CellGridLayout * const layout);
int cellGridEqual(const CellGridLayout * const layoutA, const CellGridLayout * const layoutB);
void cellGetRect(int col, int row, int* xOut, int* yOut, int* widthOut, int* heightOut);
void cellGetRectPadded(int col, int row, int padding, int* xOut, int* yOut, int* widthOut, int* heightOut);

#ifdef __cplusplus
class CellIterator2D {
private:
    int m_x, m_y, m_cellsX, m_cellsY;
public:
    CellIterator2D(int cellsX, int cellsY) : m_x(0), m_y(0), m_cellsX(cellsX), m_cellsY(cellsY) {}
    int x() const                       { return m_x; }
    int y() const                       { return m_y; }
    void setCol(int col)                { m_x = col; }
    void setRow(int row)                { m_y = row; }
    void nextCol()                      { m_x++; if (m_x >= m_cellsX) { nextRow();} }
    void nextRow()                      { m_x = 0; m_y++; }
    CellIterator2D& operator ++()       { nextCol(); return *this; }
    CellIterator2D operator ++(int)     { CellIterator2D it = *this; (*this).nextCol(); return it; }
    bool allowed() const
    {
        return (m_x < m_cellsX) && (m_y < m_cellsY) && (cellAllowed(m_x, m_y) != 0);
    }
};
#endif

#endif


