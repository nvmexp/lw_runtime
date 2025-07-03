/*
 * Copyright (c) 2006 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"
#include "cells.h"
#include "elw.h"

//////////////////////////////////////////////////////////////////////
// A mechanism leads to mouse based interactivity as a primary goal
// rows and columns start counting at 0
//

unsigned char defaultCellGrid[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL];
unsigned char interactiveCellGrid[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL];
CellGridLayout interactiveGridLayout;
CellGridLayout previousGridLayout;
int cellSelection = 0; // if the cmdline specifically asks for certain cells
int cellInteractiveSelection = 0;


void
cellGetRect(int col, int row, int* xOut, int* yOut, int* widthOut, int* heightOut)
{
    *widthOut = (lwrrentWindowWidth - interactiveGridLayout.marginLeft - interactiveGridLayout.marginRight) / interactiveGridLayout.cols;
    *heightOut = (lwrrentWindowHeight - interactiveGridLayout.marginTop - interactiveGridLayout.marginBottom) / interactiveGridLayout.rows;
    *xOut = interactiveGridLayout.marginLeft + col * *widthOut;
    *yOut = interactiveGridLayout.marginTop + row * *heightOut;
}

void cellGetRectPadded(int col, int row, int padding, int* xOut, int* yOut, int* widthOut, int* heightOut)
{
    cellGetRect(col, row, xOut, yOut, widthOut, heightOut);
    if (2*padding >= *widthOut) padding = 0;
    if (2*padding >= *heightOut) padding = 0;
    *xOut += padding;
    *yOut += padding;
    *widthOut -= padding * 2;
    *heightOut -= padding * 2;
}

void
cellInit(void)
{
    int i, j;
    for (i = 0; i < MAX_CELL_ROW_COL; i++) {
        for (j = 0; j < MAX_CELL_ROW_COL; j++) {
            defaultCellGrid[i][j] = 0;
            interactiveCellGrid[i][j] = 0;
        }
    }
}


void
cellTestInit(unsigned int cols, unsigned int rows,
             int marginLeft  /* = 0 */, int marginBottom /* = 0 */,
             int marginRight /* = 0 */, int marginTop    /* = 0 */)
{
    interactiveGridLayout.cols = cols;
    interactiveGridLayout.rows = rows;
    interactiveGridLayout.marginLeft = marginLeft;
    interactiveGridLayout.marginBottom = marginBottom;
    interactiveGridLayout.marginRight = marginRight;
    interactiveGridLayout.marginTop = marginTop;
    interactiveGridLayout.defined = 1;
}

void
cellTestFini(void)
{
    cellReset();
}

Cell
cellGet(int x, int y, const CellGridLayout * const layout)
{
    Cell cell;
    if (!layout || layout->cols == 0 || layout->rows == 0 || layout->defined == 0) {
        cell.col = 0;
        cell.row = 0;
    } else {
        int widthOut = ((lwrrentWindowWidth - layout->marginLeft - layout->marginRight) / layout->cols) * layout->cols;
        int heightOut = ((lwrrentWindowHeight - layout->marginTop - layout->marginBottom) / layout->rows) * layout->rows;

        cell.col = (x - layout->marginLeft) * ((int) layout->cols) / widthOut;
        cell.row = (lwrrentWindowHeight - y - layout->marginBottom) * ((int)layout->rows) / heightOut;
        if (cell.col >= (int)layout->cols) {
            cell.col = layout->cols - 1;
        }
        if (cell.col < 0) {
            cell.col = 0;
        }
        if (cell.row >= (int)layout->rows) {
            cell.row = layout->rows - 1;
        }
        if (cell.row < 0 ){
            cell.row = 0;
        }
    }
    return cell;
}

int
cellGridEqual(const CellGridLayout * const layoutA,
              const CellGridLayout * const layoutB)
{
    return (layoutA->defined && layoutB->defined &&
            layoutA->cols == layoutB->cols &&
            layoutA->rows == layoutB->rows &&
            layoutA->marginLeft == layoutB->marginLeft &&
            layoutA->marginBottom == layoutB->marginBottom &&
            layoutA->marginRight == layoutB->marginRight &&
            layoutA->marginTop == layoutB->marginTop
           );
}

// Must be called for all x and y in order
int
cellAllowed(int col, int row)
{
    if (cellInteractiveSelection && interactiveGridLayout.defined) {
        return interactiveCellGrid[col][row];
    } else {
        return defaultCellGrid[col][row];
    }
}

int
cellAllowedRange(int col1, int row1, int col2, int row2)
{
    int row, col;
    for (row = row1; row <= row2; row++) {
        for (col = col1; col <= col2; col++) {
            if (cellAllowed(col, row)) return 1;
        }
    }
    return 0;
}

void
cellReset(void)
{
    /* No-op for LWN */
}

int
cellInitForTest(void)
{
    return (cellSelection || cellInteractiveSelection);
}

void
cellInteractionInit(unsigned char value)
{
    int i, j;
    for (i = 0; i < MAX_CELL_ROW_COL; i++) {
        for (j = 0; j < MAX_CELL_ROW_COL; j++) {
            interactiveCellGrid[i][j] = value;
        }
    }
}

void
cellSelectRange(int col1, int row1, int col2, int row2,
                unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL])
{
    int j;
    for (j = 0; j <=   row2 * MAX_CELL_ROW_COL + col2
                        - row1 * MAX_CELL_ROW_COL - col1;
            j++)
    {
        (*grid) [ (col1 + j) % MAX_CELL_ROW_COL]
                    [row1  + (col1 + j) / MAX_CELL_ROW_COL] = 1;
    }
}

void
cellToggleRange(int col1, int row1, int col2, int row2,
                unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL])
{
    int j;
    for (j = 0; j <=   row2 * MAX_CELL_ROW_COL + col2
                        - row1 * MAX_CELL_ROW_COL - col1;
            j++)
    {
        (*grid) [ (col1 + j) % MAX_CELL_ROW_COL]
                    [row1  + (col1 + j) / MAX_CELL_ROW_COL] =
            ! (*grid) [ (col1 + j) % MAX_CELL_ROW_COL]
                                [row1  + (col1 + j) / MAX_CELL_ROW_COL];
    }
}

void
cellSelectBlock(int col1, int row1, int col2, int row2,
                unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL])
{
    int i, j;
    for (i = col1; i <= col2; i++) {
        for (j = row1; j <= row2; j++) {
            (*grid)[i][j] = 1;
        }
    }
}

void
cellToggleBlock(int col1, int row1, int col2, int row2,
                unsigned char (*grid)[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL])
{
    int i, j;
    for (i = col1; i <= col2; i++) {
        for (j = row1; j <= row2; j++) {
            (*grid)[i][j] = !(*grid)[i][j];
        }
    }
}

// Print off cmdline options that will reproduce the current selection in interactiveCellGrid and interactiveGridLayout
// First, look for an easy special case where there is a single cell block.
// If that fails, express the cells as a number of cell ranges, and use a -cellList to capture the ranges of length 1.
void
cellPrintCmdLine(void)
{
    unsigned int i, j;
    unsigned char prevCellSelected = 0;

    unsigned int startBlockCol = 0;
    unsigned int startBlockRow = 0;
    unsigned int endBlockCol = 0;
    unsigned int endBlockRow = 0;
    unsigned int blockStarted = 0;
    unsigned int blockFailed = 0;

    unsigned int startRangeCol = 0;
    unsigned int startRangeRow = 0;
    unsigned int endRangeCol = 0;
    unsigned int endRangeRow = 0;
    unsigned char cellList[MAX_CELL_ROW_COL][MAX_CELL_ROW_COL];
    unsigned int cellListNonEmpty = 0;

    // Find whether there is exactly one block
    for (j = 0; j < interactiveGridLayout.rows && !blockFailed; j++) {
        for (i = 0; i < interactiveGridLayout.cols && !blockFailed; i++)  {
            if (!blockStarted && interactiveCellGrid[i][j]) {
                // If the selection is a single block then (i,j) is the lower left corner of that block
                unsigned int ii, jj;

                blockStarted = 1;
                startBlockCol = i;
                startBlockRow = j;

                // figure out the height and width of the block (assuming that we have one)
                for (ii = i; ii < interactiveGridLayout.cols && interactiveCellGrid[ii][j]; ii++) {
                    endBlockCol = ii;
                }
                for (jj = j; jj < interactiveGridLayout.rows && interactiveCellGrid[i][jj]; jj++) {
                    endBlockRow = jj;
                }
                continue;
            }

            if (blockStarted) {
                unsigned int insidePossibleBlock =
                    (startBlockCol <= i && i <= endBlockCol && startBlockRow <= j && j <= endBlockRow);
                if ((interactiveCellGrid[i][j] && !insidePossibleBlock) ||
                    (!interactiveCellGrid[i][j] && insidePossibleBlock))
                {
                    // Either we found a cell in our block that isn't enabled, or one outside that is enabled.
                    // Clearly, the selection cannot be aclwrately described by a single -cellBlock parameter set.
                    blockFailed = 1;
                }
            }
        }
    }

    if (!blockStarted) {
        // grid is empty
        printf("Empty cell selections of cells cannot be duplicated from cmdline options. "
            "Please select one or more cells with the mouse.\n");
        return;
    }

    printf("Cell Selection Repro parameters:\n");
    if (!blockFailed) {
        // check for only a single cell
        if (startBlockCol == endBlockCol && startBlockRow == endBlockRow) {
            printf("  -cellList %d %d\n", startBlockCol, startBlockRow);
        } else {
            printf("  -cellblock %d %d %d %d\n", startBlockCol, startBlockRow, endBlockCol, endBlockRow);
        }
        return;
    }

    // The general case: use a combination of cellRange and cellList
    for (j = 0; j < interactiveGridLayout.rows; j++) {
        for (i = 0; i < interactiveGridLayout.cols; i++)  {
            // LOOP ILWARIANCE (BOTH INNER AND OUTER)
            // All ranges that END strictly prior to (i, j) have been specified by the cumulative output (or stashed away in cellList).
            // All ranges that END at or after (i, j) have not been specified by the cumulative output.
            //
            cellList[i][j] = 0;

            if (prevCellSelected) {
                if (interactiveCellGrid[i][j]) {
                    endRangeCol = i;
                    endRangeRow = j;
                    // (startRangeCol, startRangeRow) starts a range that runs at least as far as (endRangeCol, endRangeRow).
                } else {
                    if (startRangeCol == endRangeCol && startRangeRow == endRangeRow) {
                        // single-celled ranges get a special case
                        cellListNonEmpty = 1;
                        cellList[i][j] = 1;
                    } else {
                        printf("  -cellRange %d %d %d %d\n", startRangeCol, startRangeRow, endRangeCol, endRangeRow);
                    }
                    //  (i, j) and all cells prior are correctly specified by the cumulative output. (or stashed away in cellList)
                }
            } else {
                if (interactiveCellGrid[i][j]) {
                    startRangeCol = i;
                    startRangeRow = j;
                    endRangeCol = i;
                    endRangeRow = j;
                    // (startRangeCol, startRangeRow) starts a range
                } else {
                    // Do nothing

                    // We are neither starting, terminating, nor inside a range
                }
            }
            prevCellSelected = interactiveCellGrid[i][j];
        }
    }

    // There may be a outstanding range that we need to handle
    if (prevCellSelected) {
        if (startRangeCol == interactiveGridLayout.cols - 1 && startRangeRow == interactiveGridLayout.rows - 1) {
            cellList[startRangeCol][startRangeRow] = 1;
            cellListNonEmpty = 1;
        } else {
            printf("  -cellRange %d %d %d %d\n", startRangeCol, startRangeRow,
                interactiveGridLayout.cols - 1, interactiveGridLayout.rows - 1);
        }
    }

    if (cellListNonEmpty) { // collect the singletons
        // double spaces help reduce ambiguity when copying & pasting wrapped lines from console
        printf("  -cellList");
        for (j = 0; j < interactiveGridLayout.rows; j++) {
            for (i = 0; i < interactiveGridLayout.cols; i++)  {
                if (cellList[i][j]) {
                    printf("  %d  %d", i, j);
                }
            }
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

