
// sample code experimenting with how to fill in Laguna Seca's multicast/collective port list
//
// The port list hardware is organized into sets of 6 entries called rounds, each of which corresponds
// to a column in the crossbar. Entry index mod 6 indicates which column. The columns are organized in 
// two sets of even/odd pairs. The value in the entry is the relativeindex of the port within the 
// column. The columns have the following mapping between the entry and the actual port number:
//
//  Column 0, values 0..10 correspond to ports 00..10.
//  Column 1, values 0..10 correspond to ports 32..42.
//  Column 2, values 0..9  correspond to ports 11..20.
//  Column 3, values 0..10 correspond to ports 43..52.
//  Column 4, values 0..10 correspond to ports 21..31.
//  Column 5, values 0..9  correspond to ports 53..63.
//
// A value of 0x0F indicates an unused entry.
//
// The switch can process one round, or up to 6 entries, per cycle. Therefore it is advantageous to 
// minimize the number of rounds. To that end, even/odd pairs of columns can select a port from their 
// paired column.  This is controlled by the altpath element in the entry. When altpath is set to 1,
// the column to port mapping is as follows:
//
//  Column 0, values 0..10 correspond to ports 32..42.
//  Column 1, values 0..10 correspond to ports 00..10.
//  Column 2, values 0..10 correspond to ports 43..52.
//  Column 3, values 0..9  correspond to ports 11..20.
//  Column 4, values 0..9  correspond to ports 53..63.
//  Column 5, values 0..10 correspond to ports 21..31.
//


#include <stdio.h>
#include <stdlib.h>

//  Laguna port/column mappings

#define LAGUNA_PORT_00_10   0
#define LAGUNA_PORT_32_42   1
#define LAGUNA_PORT_11_20   2
#define LAGUNA_PORT_43_52   3
#define LAGUNA_PORT_21_31   4
#define LAGUNA_PORT_53_63   5

#define LAGUNA_MIN_PORT_0TH 00      // These are not in column order.
#define LAGUNA_MIN_PORT_1ST 11      // Rather they define the
#define LAGUNA_MIN_PORT_2ND 21      // points at which we go from
#define LAGUNA_MIN_PORT_3RD 32      // one column to the next
#define LAGUNA_MIN_PORT_4TH 43
#define LAGUNA_MIN_PORT_5TH 53

#define LAGUNA_PORT_COUNT               64
#define LAGUNA_ILWALID_MC_OFFSET        15
#define LAGUNA_MC_COLUMN_COUNT          6
#define LAGUNA_MC_COLUMN_PAIR_COUNT     LAGUNA_MC_COLUMN_COUNT/2
#define LAGUNA_MC_COLUMN_DEPTH          11
#define LAGUNA_MC_PLIST_SIZE            32

// for our scratch array we want to round up to the next row of column pairs.

#define LAGUNA_MC_PLIST_SCRATCH_SIZE   (LAGUNA_MC_COLUMN_DEPTH + 2) * LAGUNA_MC_COLUMN_PAIR_COUNT

typedef struct mcpentry {
    int     tcp;            // Tile column pair
    int     tcp0Port;       // Port index within even column
    bool    tcp0AltPath;    // Switch to select from odd column
    int     tcp0VCHop;      // VC selection (should be an enum)
    int     tcp1Port;       // Port index within odd column
    bool    tcp1AltPath;    // Switch to select from even column
    int     tcp1VCHop;      // VC selection (should be an enum)
    int     roundSize;      // number of tile column pairs in this round
    bool    last;           // last mcPEntry in this multicast selector
                            // could be multiple selectors in case of spray
} mcPEntry;


//
// colwert a raw port number (0..63) into a column-relative
// index.This is what goes into the port list entries
//

int hwPortField( int rawPort ) 
{ 
    if ( rawPort >= LAGUNA_PORT_COUNT )
    {
        return LAGUNA_ILWALID_MC_OFFSET;
    }
    else 
    { 
        
        if ( rawPort < LAGUNA_MIN_PORT_1ST ) 
        {
            return rawPort;
        }
        else if ( rawPort < LAGUNA_MIN_PORT_2ND )
        { 
            return rawPort - LAGUNA_MIN_PORT_1ST;
        }
        else if ( rawPort < LAGUNA_MIN_PORT_3RD )
        {
            return rawPort - LAGUNA_MIN_PORT_2ND;
        } 
        else if ( rawPort < LAGUNA_MIN_PORT_4TH )
        {
            return rawPort - LAGUNA_MIN_PORT_3RD;
        }
        else if ( rawPort < LAGUNA_MIN_PORT_5TH )
        {
            return rawPort - LAGUNA_MIN_PORT_4TH;
        }
        else
        {
            return rawPort - LAGUNA_MIN_PORT_5TH;
        }
    }
}

// This procedure prints out the port column entries of the working copy of the
// multicast port column list, first with the actual physical port numbers,
// then with the index into the column and other elements of the port list entry.

void printMcPListT( mcPEntry * mcPList )
{
    int i, j;
// loop over column pairs        
    printf(     "Raw port numbers                       mc_plist port fields\n");

    for ( i = 0; i < LAGUNA_MC_PLIST_SIZE; i += LAGUNA_MC_COLUMN_PAIR_COUNT )
    {
        printf( "Round %d, First mc_plist Index %d\n", i/3, i );
        printf( "Tile Col Pair:" );
        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("%9d ",mcPList[i + j].tcp);
        }

        printf("     ");

        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("%9d ",mcPList[i + j].tcp);
        }

        printf("\n");

        printf( "Ports:          ");
        for ( j = 0; j < LAGUNA_MC_COLUMN_COUNT/2; j++ )
        {
            printf("%3d ", mcPList[i + j].tcp0Port);
            printf("%3d   ", mcPList[i + j].tcp1Port);
        }
        printf("     ");

        for ( j = 0; j < LAGUNA_MC_COLUMN_COUNT/2; j++ )
        {
            printf("%3d ", hwPortField( mcPList[i + j].tcp0Port) );
            printf("%3d   ", hwPortField( mcPList[i + j].tcp1Port) );
        }
        printf("\n");
    
        printf( "AltPath:        ");
        for ( j = 0; j < LAGUNA_MC_COLUMN_PAIR_COUNT; j++ )
        {
            printf("%3d ", mcPList[i + j].tcp0AltPath);
            printf("%3d   ", mcPList[i + j].tcp1AltPath);
        }
        printf("     ");

        for ( j = 0; j < LAGUNA_MC_COLUMN_PAIR_COUNT; j++ )
        {
            printf("%3d ", mcPList[i + j].tcp0AltPath);
            printf("%3d   ", mcPList[i + j].tcp1AltPath);
        }
        printf("\n");


    }
    printf("\n\n");       

}

// This procedure prints out the port column entries of the final
// multicast port column list. Port numbers are indexes into their 
// respective column.

void printMcPList( mcPEntry * mcPList )
{
    int i, j;
    int round;
    bool done = false;

// loop over rounds       
    i = 0;
    round = 0;
    while ( !done && round < 20 )
    {
        printf( "Round %d, First mc_plist Index %d round size %d\n", round, i, mcPList[i].roundSize );
        printf( "Tile Col Pair:" );
        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("%9d ",mcPList[i + j].tcp);
        }
        printf("\n");

        printf( "Ports:          ");
        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("%3d ", mcPList[i + j].tcp0Port);
            printf("%3d   ", mcPList[i + j].tcp1Port);
        }
        printf("\n");
    
        printf( "AltPath:        ");
        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("%3d ", mcPList[i + j].tcp0AltPath);
            printf("%3d   ", mcPList[i + j].tcp1AltPath);
        }
        printf("\n");

        printf( "Last:           ");
        for ( j = 0; j < mcPList[i].roundSize; j++ )
        {
            printf("    %3d   ", mcPList[i + j].last);
        }
        printf("\n");
        if ( mcPList[i + j - 1].last == true )
        {
            done = true;
        } 
        i += mcPList[i].roundSize;
        round++;
    }
    printf("\n\n");       

}

// this procedure takes an array of port numbers and produces an optimized port list
// This is done in 3 steps.
// 1.Port numbers are sorted into a 2D array indexed by their column number
// 2.Ports are processed by column pairs, to generate the sets of column pairs and port offsets
//   that will participate in each round of the multicast.
// 3.The final compressed port list is produced by copying the rounds to the output array,
//   and removing the column pairs that do not participate in a given round.

// TODO this code does not yet handle spray to mulitple subsets of ports in the input 

void setupMcPList( int *portList, int spray, int *sprayCounts, mcPEntry * mcPList, int * usedCount )
{
    int i, j;
    int extraPorts;
    int nextRound;
    int nextExtra;
    int portsPerColumn[LAGUNA_MC_COLUMN_COUNT] = { 0 };
    int portsInColumn[LAGUNA_MC_COLUMN_COUNT][LAGUNA_MC_COLUMN_DEPTH] = { 0 };
    mcPEntry mcPListT[LAGUNA_MC_PLIST_SCRATCH_SIZE];    //working version prior to squeezing out empty column pairs
    int nextPair;
    int empty;


// generate a count of the number of entries in each column, and
// save the port numbers in the 2D array in order, in their respective
// columns.

    for ( i = 0; i < sprayCounts[0]; i++ )
    {
        if ( portList[i] < LAGUNA_MIN_PORT_1ST ) 
        {
            portsInColumn[LAGUNA_PORT_00_10][portsPerColumn[LAGUNA_PORT_00_10]] = portList[i];
            portsPerColumn[LAGUNA_PORT_00_10]++;
        } 
        else if ( portList[i] < LAGUNA_MIN_PORT_2ND )
        {
            portsInColumn[LAGUNA_PORT_11_20][portsPerColumn[LAGUNA_PORT_11_20]] = portList[i];
            portsPerColumn[LAGUNA_PORT_11_20]++;
        } 
        else if ( portList[i] < LAGUNA_MIN_PORT_3RD )
        {
            portsInColumn[LAGUNA_PORT_21_31][portsPerColumn[LAGUNA_PORT_21_31]] = portList[i];
            portsPerColumn[LAGUNA_PORT_21_31]++;
        } 
        else if ( portList[i] < LAGUNA_MIN_PORT_4TH )
        {
            portsInColumn[LAGUNA_PORT_32_42][portsPerColumn[LAGUNA_PORT_32_42]] = portList[i];
            portsPerColumn[LAGUNA_PORT_32_42]++;
        } 
        else if ( portList[i] < LAGUNA_MIN_PORT_5TH )
        {
            portsInColumn[LAGUNA_PORT_43_52][portsPerColumn[LAGUNA_PORT_43_52]] = portList[i];
            portsPerColumn[LAGUNA_PORT_43_52]++;
        } 
        else
        {
            portsInColumn[LAGUNA_PORT_53_63][portsPerColumn[LAGUNA_PORT_53_63]] = portList[i];
            portsPerColumn[LAGUNA_PORT_53_63]++;
        } 
    }
    for ( i = 0; i < LAGUNA_MC_COLUMN_COUNT; i++ )
    {
        printf( "%d Ports in column %d:\n",portsPerColumn[i], i );
        for ( j = 0; j < portsPerColumn[i]; j++ )
        {
            printf("%4d",portsInColumn[i][j]);
        }
        printf("\n");
    }
    printf("\n");

// Initialize the scratch port list to all invalid 

    for ( i = 0; i < LAGUNA_MC_PLIST_SCRATCH_SIZE; i ++ )
    {
        mcPListT[i].tcp = i % 3;
        mcPListT[i].roundSize = 3;
        mcPListT[i].last = false;
        mcPListT[i].tcp0Port = 255;
        mcPListT[i].tcp1Port = 255;
        mcPListT[i].tcp0AltPath = false;
        mcPListT[i].tcp1AltPath = false;
        mcPListT[i].tcp0VCHop = 0;
        mcPListT[i].tcp1VCHop = 0;
    }

// Initialize the final port list to all invalid 

    for ( i = 0; i < LAGUNA_MC_PLIST_SIZE; i ++ )
    {
        mcPList[i].tcp = 0x0F;
        mcPList[i].roundSize = 0;
        mcPList[i].last = false;
        mcPList[i].tcp0Port = 255;
        mcPList[i].tcp1Port = 255;
        mcPList[i].tcp0AltPath = false;
        mcPList[i].tcp1AltPath = false;
        mcPList[i].tcp0VCHop = 0;
        mcPList[i].tcp1VCHop = 0;
    }

// process columns pairwise. if one column is larger than the other by 2 or more entries, 
// we can use the alt path optimization

    for ( i = 0; i < LAGUNA_MC_COLUMN_PAIR_COUNT; i++ )
    {
        extraPorts = portsPerColumn[2*i] - portsPerColumn[2*i + 1];

        if ( extraPorts >= 0 )
        {
            for ( nextRound = 0; nextRound < portsPerColumn[2*i + 1]; nextRound++ )
            {
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp0Port = portsInColumn[2*i][nextRound];
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1Port = portsInColumn[2*i + 1][nextRound];
            }

            if ( extraPorts == 0 ) 
            {
                continue;
            }

            nextExtra = nextRound;
            mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp0Port = portsInColumn[2*i][nextExtra];
            extraPorts--;
            while ( extraPorts > 0 )
            {
                nextExtra++;
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1Port = portsInColumn[2*i][nextExtra];
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1AltPath = true;
                extraPorts--;
                nextRound++;
                if ( extraPorts > 0 )
                {
                    nextExtra++;
                    mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp0Port = portsInColumn[2*i][nextExtra];
                    extraPorts--;
                }
            }
        }
        else
        {
            extraPorts = -extraPorts;
            for ( nextRound = 0; nextRound < portsPerColumn[2*i]; nextRound++ )
            {
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp0Port = portsInColumn[2*i][nextRound];
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1Port = portsInColumn[2*i + 1][nextRound];
            }

            nextExtra = nextRound;
            mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1Port = portsInColumn[2*i + 1][nextExtra];
            extraPorts--;
            while ( extraPorts > 0 )
            {
                nextExtra++;
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp0Port = portsInColumn[2*i + 1][nextExtra];
                mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1AltPath = true;
                extraPorts--;
                nextRound++;
                if ( extraPorts > 0 )
                {
                    nextExtra++;
                    mcPListT[i + nextRound * LAGUNA_MC_COLUMN_PAIR_COUNT].tcp1Port = portsInColumn[2*i + 1][nextExtra];
                    extraPorts--;
                }
            }
        }
    }

    printMcPListT( mcPListT );

// the scratch port list has the raw port numbers we want. These have to be colwerted to
// their crossbar column relative indexes.

    for ( i = 0; i < LAGUNA_MC_PLIST_SCRATCH_SIZE; i++ )
    {
        mcPListT[i].tcp0Port = hwPortField( mcPListT[i].tcp0Port );
        mcPListT[i].tcp1Port = hwPortField( mcPListT[i].tcp1Port );
    }

// finally we need to squeeze out the port column pairs that we are not using, 
// adjust the size of the round, and set the last flag
    
    nextPair = 0;

    for ( i = 0; i < LAGUNA_MC_PLIST_SCRATCH_SIZE; i += LAGUNA_MC_COLUMN_PAIR_COUNT )
    {
//Count the number of empty pairs in this round
        empty = 0;
        for ( j = 0; j < LAGUNA_MC_COLUMN_PAIR_COUNT; j++ )
        {
            if ( mcPListT[i + j].tcp0Port == 0x0F && mcPListT[i + j].tcp1Port == 0x0F )
            {
                empty++;
            }
        }
        if ( empty == LAGUNA_MC_COLUMN_PAIR_COUNT )
        {
            break;  //no more entries to process after an empty round
        }

//Set the round size in the next pair entry and copy used entries
        mcPList[nextPair].roundSize = LAGUNA_MC_COLUMN_PAIR_COUNT - empty;

        for ( j = 0; j < LAGUNA_MC_COLUMN_PAIR_COUNT; j++ )
        {
            if ( !( mcPListT[i + j].tcp0Port == 0x0F && mcPListT[i + j].tcp1Port == 0x0F ) )
            {
                mcPList[nextPair].tcp = mcPListT[i + j].tcp;
                mcPList[nextPair].tcp0Port = mcPListT[i + j].tcp0Port;
                mcPList[nextPair].tcp0AltPath = mcPListT[i + j].tcp0AltPath;
                mcPList[nextPair].tcp0VCHop = mcPListT[i + j].tcp0VCHop;
                mcPList[nextPair].tcp1Port = mcPListT[i + j].tcp1Port;
                mcPList[nextPair].tcp1AltPath = mcPListT[i + j].tcp1AltPath;
                mcPList[nextPair].tcp1VCHop = mcPListT[i + j].tcp1VCHop;
                nextPair++;
            }
        }
    }
//Set the last flag for the last entry in the spray string.
    mcPList[nextPair -1].last = true;
    *usedCount = nextPair;

}

main()
{
    int i,j;
    int select;
    int portsInGroup;
    int count = 0;
    bool selected[LAGUNA_PORT_COUNT];
    int selectPort[LAGUNA_PORT_COUNT];
    int selectCount;
    int sprayCounts[16];
    mcPEntry mcPList[LAGUNA_MC_PLIST_SIZE];
    int usedCount;
    int selectedPerColumn[LAGUNA_MC_COLUMN_COUNT];
    int randDiv = RAND_MAX/LAGUNA_PORT_COUNT;
    
    // try out a few special cases

    portsInGroup = 64;
    for ( i = 0; i < LAGUNA_PORT_COUNT; i++ )
    {
        selectPort[i] = i;
    }
    printf( "\n" ) ;

    setupMcPList( selectPort, 1, &portsInGroup, mcPList, &usedCount );
    printMcPList( mcPList );

    portsInGroup = 32;    
    for ( i = 0; i < LAGUNA_PORT_COUNT/2; i++ )
    {
        selectPort[i] = i;
    }
    printf( "\n" ) ;

    setupMcPList( selectPort, 1, &portsInGroup, mcPList, &usedCount );
    printMcPList( mcPList );
    
    for ( i = 0; i < LAGUNA_PORT_COUNT/2; i++ )
    {
        selectPort[i] = i + LAGUNA_PORT_COUNT/2;
    }
    printf( "\n" ) ;

    setupMcPList( selectPort, 1, &portsInGroup, mcPList, &usedCount );
    printMcPList( mcPList );
    

    // warm up rand
    for ( i = 0; i < 1000; i++ )
    {
        j = rand();
    }

    // loop over some number of random permutations
    while ( count < 100 )
    {
        selectCount = 0;
        for ( i = 0; i < LAGUNA_PORT_COUNT; i++ )
        {
            selected[i] = false;
        }
        while ( selectCount < portsInGroup )
        {
            select = rand()/randDiv;
            if ( selected[select] )
            {
                continue;
            } else {
                selected[select] = true;
                selectCount++;
            }
        }

		printf( "count %d\n", count );
        printf( "selected ports:\n");
        selectCount = 0;

        for ( i = 0; i < LAGUNA_PORT_COUNT; i += 8 )
        {
             for ( j= 0; j < 8; j++ )
             {
                if ( selected[ i + j ] )
                {    
                    selectPort[selectCount] = i + j;                    
                    printf( "%4d", i + j );
                    selectCount++;
                    if ( selectCount % 8 == 0 )
                    {
                        printf( "\n" );
                    }
                }
            }
        } 
                       
        printf( "\n" ) ;
        setupMcPList( selectPort, 1, &portsInGroup, mcPList, &usedCount );

        printMcPList( mcPList );

        count++;
        
    }        

}
