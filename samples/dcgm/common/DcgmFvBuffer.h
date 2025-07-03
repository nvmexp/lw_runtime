#ifndef DCGMFVBUFFER_H
#define DCGMFVBUFFER_H

#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <stddef.h> //size_t

/**
 * This structure is used to buffer field values within DCGM, so it is optimized for storage size.
 * Fields like status, fieldType, and entityGroup are smaller than their source types because 
 * their values are small enums. 
 * 
 * Each field should start on a multiple of its size. For instance, timestamp is 8 bytes and 
 * starts at offset 16.
 * 
 * This structure starts with a length so that the buffer can be walked regardless
 * of understanding a specific version of this struct.
 */
typedef struct
{
    unsigned short length;              //!< Length of this entry. 
    unsigned char version;              //!< version number (dcgmBufferedFv_version)
    unsigned char fieldType;            //!< One of DCGM_FT_?
    signed char status;                 //!< Status for the querying the field. DCGM_ST_OK or one of DCGM_ST_?
    unsigned char entityGroupId;        //!< Entity group this field value belongs to (dcgm_field_entity_group_t)
    unsigned short fieldId;             //!<  One of DCGM_FI_?
    int64_t timestamp;                  //!< Timestamp in usec since 1970 
    dcgm_field_eid_t entityId;          //!< Entity this field value belongs to
    unsigned int unused;                //!< Unused space to align .value to a 8-byte boundary. Set to 0 */
    /* 24 bytes to here */

    /* This union should come last, as length will be adjusted to truncate this based on 
       which fieldType this is */
    union {
        int64_t i64;      //!<  Int64 value
        double  dbl;      //!< Double value
        char    str[DCGM_MAX_STR_LENGTH]; //!< NULL terminated string
        char    blob[DCGM_MAX_BLOB_LENGTH]; //!< Binary blob
    } value;            //!< Value
} dcgmBufferedFv_v1, dcgmBufferedFv_t;

/* Current version of dcgmBufferedFv_t */
#define dcgmBufferedFv_version 1

/* Represents a cursor into a FV buffer */
typedef size_t dcgmBufferedFvLwrsor_t;

/* Macro to estimate the initial capacity needed */
#define FVBUFFER_GUESS_INITIAL_CAPACITY(numEntities, numFieldIds) \
        ((size_t)((numEntities)*(numFieldIds)*32))

/*
 * DcgmFvBuffer is a class for buffering field values between subsystems and processes
 * with minimal thrashing of the heap.
 * 
 * This class is not designed to be thread safe. You should pass this instance between
 * threads or make a copy of it if you need to use it conlwrrently. 
 */

class DcgmFvBuffer 
{
public:
    /**************************************************************************
     * Constructor
     * 
     * initialCapacity  IN: Initial capacity of this structure in bytes. The minimum
     *                      size of a dcgmBufferedFv_v1 is 32 bytes, so you should
     *                      set this to 32 * how many records you expect to buffer.
     *                      You can use FVBUFFER_GUESS_INITIAL_CAPAXITY to get this
     *                      number for you
     */
    DcgmFvBuffer(size_t initialCapacity = 512);
    
    /*************************************************************************/
    ~DcgmFvBuffer();

    /**************************************************************************
     * Clear the contents of this structure
     * 
     */
    void Clear(void);

    /**************************************************************************
     * Get the next field value in this structure based on the passed-in cursor
     *
     * cursor IN/OUT: Cursor from a previous call to GetNextFv(). Pass 0 on first call.
     * 
     * Returns Pointer to the next element in the buffered FV
     *         NULL if we have walked the entire FV buffer or an error oclwrs
     */
    dcgmBufferedFv_t *GetNextFv(dcgmBufferedFvLwrsor_t *cursor);

    /**************************************************************************
     * Set the contents of this FV from a buffer. This is essentially
     * deserialization
     * 
     * Returns DCGM_ST_OK on success
     *         DCGM_ST_? #define on error
     */
    dcgmReturn_t SetFromBuffer(const char *buffer, size_t bufferSize);

    /**************************************************************************
     * Get the current size of this structure in bytes and elements
     * 
     * Returns DCGM_ST_OK on success
     *         DCGM_ST_? #define on error
     */
    dcgmReturn_t GetSize(size_t *bufferSize, size_t *elementCount);

    /**************************************************************************
     *
     * Get a pointer to this fvBuffer's internal buffer. This is for serialization
     * only. Do not modify this buffer. This value may be NULL.
     *
     */
    const char *GetBuffer(void) { return m_buffer; }

    /**************************************************************************
     * Helper method to colwert a buffered FV to a FV version 1
     * 
     * Returns Nothing.
     */
    static void ColwertBufferedFvToFv1(dcgmBufferedFv_t *fv, dcgmFieldValue_v1 *fv1);

    /**************************************************************************
     * Helper method to colwert a buffered FV to a FV version 2
     * 
     * Returns Nothing.
     */
    static void ColwertBufferedFvToFv2(dcgmBufferedFv_t *fv, dcgmFieldValue_v2 *fv2);

    /**************************************************************************
     * Helper to colwert this entire structure to an array of FV version 1s
     *
     * fv1         OUT: Buffer to hold FV1s
     * fv1Capacity  IN: How many entries fv1 can hold
     * numStored   OUT: How many entries were saved to fv1. Pass NULL if you don't care.
     *
     * Returns: DCGM_ST_OK on success
     *          Other DCGM_ST_? #define on error
     *
     */
    dcgmReturn_t GetAllAsFv1(dcgmFieldValue_v1 *fv1, size_t fv1Capacity, size_t *numStored);

    /**************************************************************************
     * Allocate space for a new field value on this structure and return a pointer
     * to it. Note that this effectively adds to the structure and cannot be undone 
     * 
     * Returns A pointer to the allocated field-value structure. Do NOT zero
     *             this structure as all of its fields have been set before being returned
     *         NULL on error. This is most likely because of being out of memory.
     */
    dcgmBufferedFv_t *AddInt64Value(dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId, unsigned short fieldId, 
                                    long long value, long long timestamp, 
                                    dcgmReturn_t status);
    dcgmBufferedFv_t *AddDoubleValue(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId, unsigned short fieldId, 
                                     double value, long long timestamp, 
                                     dcgmReturn_t status);
    dcgmBufferedFv_t *AddStringValue(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId, unsigned short fieldId, 
                                     char *value, long long timestamp, 
                                     dcgmReturn_t status);
    dcgmBufferedFv_t *AddBlobValue(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId, unsigned short fieldId, 
                                   void *value, size_t valueSize, long long timestamp, 
                                   dcgmReturn_t status);

private:
    /**************************************************************************
     * Resize this structure to hold a new capacity of bytes
     * 
     * newCapacity  IN: What the new capacity of this structure should be in bytes.
     * 
     * Returns: DCGM_ST_OK on success
     *          DCGM_ST_MEMORY if we're out of memory
     */
    dcgmReturn_t Resize(size_t newCapacity);

    /**************************************************************************
     * Really add a FV to this buffer and return a pointer to it
     * 
     * bytesNeeded  IN: How many bytes are needed by this new FV
     * 
     * Returns: DCGM_ST_OK on success
     *          NULL on error. Likely out of memory
     */
    dcgmBufferedFv_t *AddFvReally(size_t bytesNeeded);
    
    char *m_buffer; /* Buffer of FVs, one after another. m_bufferUsed is how many bytes
                       of this are used. m_bufferCapacity is how many bytes this buffer
                       can hold before being resized. */
    size_t m_bufferUsed;     /* How much of this buffer is lwrrently used in bytes */
    size_t m_bufferCapacity; /* how many bytes this buffer can hold before being resized */
    size_t m_numEntries;     /* Number of FVs that are lwrrently buffered */
};

#endif //DCGMFVBUFFER_H

