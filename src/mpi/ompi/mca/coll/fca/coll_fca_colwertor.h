#ifndef MCA_COLL_FCA_COLWERTOR_H
#define MCA_COLL_FCA_COLWERTOR_H




enum {
    MCA_COLL_COLWERTOR_NULL = 0,
    MCA_COLL_FCA_COLW_SEND,
    MCA_COLL_FCA_COLW_RECV
};


struct mca_coll_fca_colwertor {
    int               type;
    FCA_COLWERTOR_T  ompic;
    size_t            size;
    void              *buf;
};

#define MCA_COLL_FCA_DECLARE_COLWERTOR(__name) \
    struct mca_coll_fca_colwertor __name = {MCA_COLL_COLWERTOR_NULL}


static inline void mca_coll_fca_colwertor_set(struct mca_coll_fca_colwertor *colw,
                                              struct ompi_datatype_t *datatype,
                                              void *buffer, int count)
{
    if (colw->type == MCA_COLL_FCA_COLW_SEND) {
        FCA_COLWERTOR_COPY_AND_PREPARE_FOR_SEND(ompi_mpi_local_colwertor,
                                                 &datatype->super, count,
                                                 buffer, 0, &colw->ompic);
    } else if (colw->type == MCA_COLL_FCA_COLW_RECV) {
        FCA_COLWERTOR_COPY_AND_PREPARE_FOR_RECV(ompi_mpi_local_colwertor,
                                                 &datatype->super, count,
                                                 buffer, 0, &colw->ompic);
    }
}

static inline void mca_coll_fca_colwertor_create(struct mca_coll_fca_colwertor *colw,
                                                 struct ompi_datatype_t *datatype,
                                                 int count, void *buffer, int type,
                                                 void **tmpbuf, size_t *size)
{
    OBJ_CONSTRUCT(&colw->ompic, FCA_COLWERTOR_T);
    colw->type = type;
    mca_coll_fca_colwertor_set(colw, datatype, buffer, count);
    FCA_COLWERTOR_COLWERTOR_GET_PACKED_SIZE(&colw->ompic, &colw->size);
    colw->buf = malloc(colw->size);
    *tmpbuf = colw->buf;
    *size = colw->size;
}

static inline int mca_coll_fca_colwertor_valid(struct mca_coll_fca_colwertor *colw)
{
    return colw->type != MCA_COLL_COLWERTOR_NULL;
}

static inline void mca_coll_fca_colwertor_destroy(struct mca_coll_fca_colwertor *colw)
{
    if (mca_coll_fca_colwertor_valid(colw)) {
        free(colw->buf);
        OBJ_DESTRUCT(&colw->ompic);
    }
}

static inline int32_t mca_coll_fca_colwertor_process(struct mca_coll_fca_colwertor *colw,
                                                     size_t offset)
{
    struct iovec ilwec;
    unsigned iov_count;
    size_t size;

    iov_count = 1;
    ilwec.iov_base = (char*)colw->buf + offset;
    ilwec.iov_len = colw->size;
    size = colw->size;

    if (colw->type == MCA_COLL_FCA_COLW_SEND) {
        return FCA_COLWERTOR_COLWERTOR_PACK(&colw->ompic, &ilwec, &iov_count, &size);
    } else if (colw->type == MCA_COLL_FCA_COLW_RECV) {
        return FCA_COLWERTOR_COLWERTOR_UNPACK(&colw->ompic, &ilwec, &iov_count, &size);
    }
    return 0;
}
#endif
