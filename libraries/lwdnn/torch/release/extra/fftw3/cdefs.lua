-- Cut and paste from the C preprocessor output
-- Removed inline/defined functions which are not supported by luajit
-- Instead, those are defined into defines.lua
-- Note there are some tests here and there to stay cross-platform

local ffi = require 'ffi'

ffi.cdef[[
typedef struct _FILE FILE;
typedef long double __float128;
]]


ffi.cdef[[

enum fftw_r2r_kind_do_not_use_me {
     FFTW_R2HC=0, FFTW_HC2R=1, FFTW_DHT=2,
     FFTW_REDFT00=3, FFTW_REDFT01=4, FFTW_REDFT10=5, FFTW_REDFT11=6,
     FFTW_RODFT00=7, FFTW_RODFT01=8, FFTW_RODFT10=9, FFTW_RODFT11=10
};

struct fftw_iodim_do_not_use_me {
     int n;
     int is;
     int os;
};

struct fftw_iodim64_do_not_use_me {
     ptrdiff_t n;
     ptrdiff_t is;
     ptrdiff_t os;
};

typedef void (*fftw_write_char_func_do_not_use_me)(char c, void *);
typedef int (*fftw_read_char_func_do_not_use_me)(void *);

typedef double fftw_complex[2]; 
typedef struct fftw_plan_s *fftw_plan; 
typedef struct fftw_iodim_do_not_use_me fftw_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftw_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftw_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftw_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftw_read_char_func; 
extern void fftw_exelwte(const fftw_plan p); 
extern fftw_plan fftw_plan_dft(int rank, const int *n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
extern fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags); 
extern fftw_plan fftw_plan_guru64_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags); 
extern fftw_plan fftw_plan_guru64_split_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags); 
extern void fftw_exelwte_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out); 
extern void fftw_exelwte_split_dft(const fftw_plan p, double *ri, double *ii, double *ro, double *io); 
extern fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftw_plan fftw_plan_dft_r2c(int rank, const int *n, double *in, fftw_complex *out, unsigned flags); 
extern fftw_plan fftw_plan_dft_r2c_1d(int n,double *in,fftw_complex *out,unsigned flags); 
extern fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags); 
extern fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double *in, fftw_complex *out, unsigned flags); 
extern fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags); 
extern fftw_plan fftw_plan_dft_c2r_1d(int n,fftw_complex *in,double *out,unsigned flags); 
extern fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out, unsigned flags); 
extern fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex *in, double *out, unsigned flags); 
extern fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, fftw_complex *out, unsigned flags); 
extern fftw_plan fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, double *out, unsigned flags); 
extern fftw_plan fftw_plan_guru_split_dft_r2c( int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *ro, double *io, unsigned flags); 
extern fftw_plan fftw_plan_guru_split_dft_c2r( int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *out, unsigned flags); 
extern fftw_plan fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, fftw_complex *out, unsigned flags); 
extern fftw_plan fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, double *out, unsigned flags); 
extern fftw_plan fftw_plan_guru64_split_dft_r2c( int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *ro, double *io, unsigned flags); 
extern fftw_plan fftw_plan_guru64_split_dft_c2r( int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *out, unsigned flags); 
extern void fftw_exelwte_dft_r2c(const fftw_plan p, double *in, fftw_complex *out); 
extern void fftw_exelwte_dft_c2r(const fftw_plan p, fftw_complex *in, double *out); 
extern void fftw_exelwte_split_dft_r2c(const fftw_plan p, double *in, double *ro, double *io); 
extern void fftw_exelwte_split_dft_c2r(const fftw_plan p, double *ri, double *ii, double *out); 
extern fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags); 
extern fftw_plan fftw_plan_r2r(int rank, const int *n, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
extern fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags); 
extern fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags); 
extern fftw_plan fftw_plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags); 
extern fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
extern fftw_plan fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags); 
extern void fftw_exelwte_r2r(const fftw_plan p, double *in, double *out); 
extern void fftw_destroy_plan(fftw_plan p); 
extern void fftw_forget_wisdom(void); 
extern void fftw_cleanup(void); 
extern void fftw_set_timelimit(double t); 
extern void fftw_plan_with_nthreads(int nthreads); 
extern int fftw_init_threads(void); 
extern void fftw_cleanup_threads(void); 
extern int fftw_export_wisdom_to_filename(const char *filename); 
extern void fftw_export_wisdom_to_file(FILE *output_file); 
extern char *fftw_export_wisdom_to_string(void); 
extern void fftw_export_wisdom(fftw_write_char_func write_char, void *data); 
extern int fftw_import_system_wisdom(void); 
extern int fftw_import_wisdom_from_filename(const char *filename); 
extern int fftw_import_wisdom_from_file(FILE *input_file); 
extern int fftw_import_wisdom_from_string(const char *input_string); 
extern int fftw_import_wisdom(fftw_read_char_func read_char, void *data); 
extern void fftw_fprint_plan(const fftw_plan p, FILE *output_file); 
extern void fftw_print_plan(const fftw_plan p); 
extern void *fftw_malloc(size_t n); 
extern double *fftw_alloc_real(size_t n); 
extern fftw_complex *fftw_alloc_complex(size_t n); 
extern void fftw_free(void *p); 
extern void fftw_flops(const fftw_plan p, double *add, double *mul, double *fmas); 
extern double fftw_estimate_cost(const fftw_plan p); 
extern double fftw_cost(const fftw_plan p); 
extern const char fftw_version[]; 
extern const char fftw_cc[]; 
extern const char fftw_codelet_optim[];
typedef float fftwf_complex[2]; 
typedef struct fftwf_plan_s *fftwf_plan; 
typedef struct fftw_iodim_do_not_use_me fftwf_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftwf_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftwf_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftwf_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftwf_read_char_func; 
extern void fftwf_exelwte(const fftwf_plan p); 
extern fftwf_plan fftwf_plan_dft(int rank, const int *n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_split_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_split_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags); 
extern void fftwf_exelwte_dft(const fftwf_plan p, fftwf_complex *in, fftwf_complex *out); 
extern void fftwf_exelwte_split_dft(const fftwf_plan p, float *ri, float *ii, float *ro, float *io); 
extern fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in, fftwf_complex *out, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_r2c_1d(int n,float *in,fftwf_complex *out,unsigned flags); 
extern fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float *in, fftwf_complex *out, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float *in, fftwf_complex *out, unsigned flags); 
extern fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_c2r(int rank, const int *n, fftwf_complex *in, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_c2r_1d(int n,fftwf_complex *in,float *out,unsigned flags); 
extern fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex *in, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, fftwf_complex *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_split_dft_r2c( int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *ro, float *io, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_split_dft_c2r( int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, fftwf_complex *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, float *out, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_split_dft_r2c( int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *ro, float *io, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_split_dft_c2r( int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *out, unsigned flags); 
extern void fftwf_exelwte_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out); 
extern void fftwf_exelwte_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out); 
extern void fftwf_exelwte_split_dft_r2c(const fftwf_plan p, float *in, float *ro, float *io); 
extern void fftwf_exelwte_split_dft_c2r(const fftwf_plan p, float *ri, float *ii, float *out); 
extern fftwf_plan fftwf_plan_many_r2r(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags); 
extern fftwf_plan fftwf_plan_r2r(int rank, const int *n, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
extern fftwf_plan fftwf_plan_r2r_1d(int n, float *in, float *out, fftwf_r2r_kind kind, unsigned flags); 
extern fftwf_plan fftwf_plan_r2r_2d(int n0, int n1, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, unsigned flags); 
extern fftwf_plan fftwf_plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, fftwf_r2r_kind kind2, unsigned flags); 
extern fftwf_plan fftwf_plan_guru_r2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
extern fftwf_plan fftwf_plan_guru64_r2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags); 
extern void fftwf_exelwte_r2r(const fftwf_plan p, float *in, float *out); 
extern void fftwf_destroy_plan(fftwf_plan p); 
extern void fftwf_forget_wisdom(void); 
extern void fftwf_cleanup(void); 
extern void fftwf_set_timelimit(double t); 
extern void fftwf_plan_with_nthreads(int nthreads); 
extern int fftwf_init_threads(void); 
extern void fftwf_cleanup_threads(void); 
extern int fftwf_export_wisdom_to_filename(const char *filename); 
extern void fftwf_export_wisdom_to_file(FILE *output_file); 
extern char *fftwf_export_wisdom_to_string(void); 
extern void fftwf_export_wisdom(fftwf_write_char_func write_char, void *data); 
extern int fftwf_import_system_wisdom(void); 
extern int fftwf_import_wisdom_from_filename(const char *filename); 
extern int fftwf_import_wisdom_from_file(FILE *input_file); 
extern int fftwf_import_wisdom_from_string(const char *input_string); 
extern int fftwf_import_wisdom(fftwf_read_char_func read_char, void *data); 
extern void fftwf_fprint_plan(const fftwf_plan p, FILE *output_file); 
extern void fftwf_print_plan(const fftwf_plan p); 
extern void *fftwf_malloc(size_t n); 
extern float *fftwf_alloc_real(size_t n); 
extern fftwf_complex *fftwf_alloc_complex(size_t n); 
extern void fftwf_free(void *p); 
extern void fftwf_flops(const fftwf_plan p, double *add, double *mul, double *fmas); 
extern double fftwf_estimate_cost(const fftwf_plan p); 
extern double fftwf_cost(const fftwf_plan p); 
extern const char fftwf_version[]; 
extern const char fftwf_cc[]; 
extern const char fftwf_codelet_optim[];
typedef long double fftwl_complex[2]; 
typedef struct fftwl_plan_s *fftwl_plan; 
typedef struct fftw_iodim_do_not_use_me fftwl_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftwl_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftwl_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftwl_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftwl_read_char_func; 
extern void fftwl_exelwte(const fftwl_plan p); 
extern fftwl_plan fftwl_plan_dft(int rank, const int *n, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_1d(int n, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_2d(int n0, int n1, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_3d(int n0, int n1, int n2, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_many_dft(int rank, const int *n, int howmany, fftwl_complex *in, const int *inembed, int istride, int idist, fftwl_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_dft(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_split_dft(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *ri, long double *ii, long double *ro, long double *io, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_dft(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_split_dft(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *ri, long double *ii, long double *ro, long double *io, unsigned flags); 
extern void fftwl_exelwte_dft(const fftwl_plan p, fftwl_complex *in, fftwl_complex *out); 
extern void fftwl_exelwte_split_dft(const fftwl_plan p, long double *ri, long double *ii, long double *ro, long double *io); 
extern fftwl_plan fftwl_plan_many_dft_r2c(int rank, const int *n, int howmany, long double *in, const int *inembed, int istride, int idist, fftwl_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_r2c(int rank, const int *n, long double *in, fftwl_complex *out, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_r2c_1d(int n,long double *in,fftwl_complex *out,unsigned flags); 
extern fftwl_plan fftwl_plan_dft_r2c_2d(int n0, int n1, long double *in, fftwl_complex *out, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_r2c_3d(int n0, int n1, int n2, long double *in, fftwl_complex *out, unsigned flags); 
extern fftwl_plan fftwl_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwl_complex *in, const int *inembed, int istride, int idist, long double *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_c2r(int rank, const int *n, fftwl_complex *in, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_c2r_1d(int n,fftwl_complex *in,long double *out,unsigned flags); 
extern fftwl_plan fftwl_plan_dft_c2r_2d(int n0, int n1, fftwl_complex *in, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_dft_c2r_3d(int n0, int n1, int n2, fftwl_complex *in, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_dft_r2c(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, fftwl_complex *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_dft_c2r(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, fftwl_complex *in, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_split_dft_r2c( int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, long double *ro, long double *io, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_split_dft_c2r( int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *ri, long double *ii, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_dft_r2c(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, fftwl_complex *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_dft_c2r(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, fftwl_complex *in, long double *out, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_split_dft_r2c( int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, long double *ro, long double *io, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_split_dft_c2r( int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *ri, long double *ii, long double *out, unsigned flags); 
extern void fftwl_exelwte_dft_r2c(const fftwl_plan p, long double *in, fftwl_complex *out); 
extern void fftwl_exelwte_dft_c2r(const fftwl_plan p, fftwl_complex *in, long double *out); 
extern void fftwl_exelwte_split_dft_r2c(const fftwl_plan p, long double *in, long double *ro, long double *io); 
extern void fftwl_exelwte_split_dft_c2r(const fftwl_plan p, long double *ri, long double *ii, long double *out); 
extern fftwl_plan fftwl_plan_many_r2r(int rank, const int *n, int howmany, long double *in, const int *inembed, int istride, int idist, long double *out, const int *onembed, int ostride, int odist, const fftwl_r2r_kind *kind, unsigned flags); 
extern fftwl_plan fftwl_plan_r2r(int rank, const int *n, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags); 
extern fftwl_plan fftwl_plan_r2r_1d(int n, long double *in, long double *out, fftwl_r2r_kind kind, unsigned flags); 
extern fftwl_plan fftwl_plan_r2r_2d(int n0, int n1, long double *in, long double *out, fftwl_r2r_kind kind0, fftwl_r2r_kind kind1, unsigned flags); 
extern fftwl_plan fftwl_plan_r2r_3d(int n0, int n1, int n2, long double *in, long double *out, fftwl_r2r_kind kind0, fftwl_r2r_kind kind1, fftwl_r2r_kind kind2, unsigned flags); 
extern fftwl_plan fftwl_plan_guru_r2r(int rank, const fftwl_iodim *dims, int howmany_rank, const fftwl_iodim *howmany_dims, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags); 
extern fftwl_plan fftwl_plan_guru64_r2r(int rank, const fftwl_iodim64 *dims, int howmany_rank, const fftwl_iodim64 *howmany_dims, long double *in, long double *out, const fftwl_r2r_kind *kind, unsigned flags); 
extern void fftwl_exelwte_r2r(const fftwl_plan p, long double *in, long double *out); 
extern void fftwl_destroy_plan(fftwl_plan p); 
extern void fftwl_forget_wisdom(void); 
extern void fftwl_cleanup(void); 
extern void fftwl_set_timelimit(double t); 
extern void fftwl_plan_with_nthreads(int nthreads); 
extern int fftwl_init_threads(void); 
extern void fftwl_cleanup_threads(void); 
extern int fftwl_export_wisdom_to_filename(const char *filename); 
extern void fftwl_export_wisdom_to_file(FILE *output_file); 
extern char *fftwl_export_wisdom_to_string(void); 
extern void fftwl_export_wisdom(fftwl_write_char_func write_char, void *data); 
extern int fftwl_import_system_wisdom(void); 
extern int fftwl_import_wisdom_from_filename(const char *filename); 
extern int fftwl_import_wisdom_from_file(FILE *input_file); 
extern int fftwl_import_wisdom_from_string(const char *input_string); 
extern int fftwl_import_wisdom(fftwl_read_char_func read_char, void *data); 
extern void fftwl_fprint_plan(const fftwl_plan p, FILE *output_file); 
extern void fftwl_print_plan(const fftwl_plan p); 
extern void *fftwl_malloc(size_t n); 
extern long double *fftwl_alloc_real(size_t n); 
extern fftwl_complex *fftwl_alloc_complex(size_t n); 
extern void fftwl_free(void *p); 
extern void fftwl_flops(const fftwl_plan p, double *add, double *mul, double *fmas); 
extern double fftwl_estimate_cost(const fftwl_plan p); 
extern double fftwl_cost(const fftwl_plan p); 
extern const char fftwl_version[]; 
extern const char fftwl_cc[]; 
extern const char fftwl_codelet_optim[];
typedef __float128 fftwq_complex[2]; 
typedef struct fftwq_plan_s *fftwq_plan; 
typedef struct fftw_iodim_do_not_use_me fftwq_iodim; 
typedef struct fftw_iodim64_do_not_use_me fftwq_iodim64; 
typedef enum fftw_r2r_kind_do_not_use_me fftwq_r2r_kind; 
typedef fftw_write_char_func_do_not_use_me fftwq_write_char_func; 
typedef fftw_read_char_func_do_not_use_me fftwq_read_char_func; 
extern void fftwq_exelwte(const fftwq_plan p); 
extern fftwq_plan fftwq_plan_dft(int rank, const int *n, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_1d(int n, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_2d(int n0, int n1, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_3d(int n0, int n1, int n2, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_many_dft(int rank, const int *n, int howmany, fftwq_complex *in, const int *inembed, int istride, int idist, fftwq_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_dft(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_split_dft(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_dft(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, fftwq_complex *in, fftwq_complex *out, int sign, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_split_dft(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io, unsigned flags); 
extern void fftwq_exelwte_dft(const fftwq_plan p, fftwq_complex *in, fftwq_complex *out); 
extern void fftwq_exelwte_split_dft(const fftwq_plan p, __float128 *ri, __float128 *ii, __float128 *ro, __float128 *io); 
extern fftwq_plan fftwq_plan_many_dft_r2c(int rank, const int *n, int howmany, __float128 *in, const int *inembed, int istride, int idist, fftwq_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_r2c(int rank, const int *n, __float128 *in, fftwq_complex *out, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_r2c_1d(int n,__float128 *in,fftwq_complex *out,unsigned flags); 
extern fftwq_plan fftwq_plan_dft_r2c_2d(int n0, int n1, __float128 *in, fftwq_complex *out, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_r2c_3d(int n0, int n1, int n2, __float128 *in, fftwq_complex *out, unsigned flags); 
extern fftwq_plan fftwq_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwq_complex *in, const int *inembed, int istride, int idist, __float128 *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_c2r(int rank, const int *n, fftwq_complex *in, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_c2r_1d(int n,fftwq_complex *in,__float128 *out,unsigned flags); 
extern fftwq_plan fftwq_plan_dft_c2r_2d(int n0, int n1, fftwq_complex *in, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_dft_c2r_3d(int n0, int n1, int n2, fftwq_complex *in, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_dft_r2c(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, fftwq_complex *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_dft_c2r(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, fftwq_complex *in, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_split_dft_r2c( int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, __float128 *ro, __float128 *io, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_split_dft_c2r( int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *ri, __float128 *ii, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_dft_r2c(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, fftwq_complex *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_dft_c2r(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, fftwq_complex *in, __float128 *out, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_split_dft_r2c( int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, __float128 *ro, __float128 *io, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_split_dft_c2r( int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *ri, __float128 *ii, __float128 *out, unsigned flags); 
extern void fftwq_exelwte_dft_r2c(const fftwq_plan p, __float128 *in, fftwq_complex *out); 
extern void fftwq_exelwte_dft_c2r(const fftwq_plan p, fftwq_complex *in, __float128 *out); 
extern void fftwq_exelwte_split_dft_r2c(const fftwq_plan p, __float128 *in, __float128 *ro, __float128 *io); 
extern void fftwq_exelwte_split_dft_c2r(const fftwq_plan p, __float128 *ri, __float128 *ii, __float128 *out); 
extern fftwq_plan fftwq_plan_many_r2r(int rank, const int *n, int howmany, __float128 *in, const int *inembed, int istride, int idist, __float128 *out, const int *onembed, int ostride, int odist, const fftwq_r2r_kind *kind, unsigned flags); 
extern fftwq_plan fftwq_plan_r2r(int rank, const int *n, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags); 
extern fftwq_plan fftwq_plan_r2r_1d(int n, __float128 *in, __float128 *out, fftwq_r2r_kind kind, unsigned flags); 
extern fftwq_plan fftwq_plan_r2r_2d(int n0, int n1, __float128 *in, __float128 *out, fftwq_r2r_kind kind0, fftwq_r2r_kind kind1, unsigned flags); 
extern fftwq_plan fftwq_plan_r2r_3d(int n0, int n1, int n2, __float128 *in, __float128 *out, fftwq_r2r_kind kind0, fftwq_r2r_kind kind1, fftwq_r2r_kind kind2, unsigned flags); 
extern fftwq_plan fftwq_plan_guru_r2r(int rank, const fftwq_iodim *dims, int howmany_rank, const fftwq_iodim *howmany_dims, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags); 
extern fftwq_plan fftwq_plan_guru64_r2r(int rank, const fftwq_iodim64 *dims, int howmany_rank, const fftwq_iodim64 *howmany_dims, __float128 *in, __float128 *out, const fftwq_r2r_kind *kind, unsigned flags); 
extern void fftwq_exelwte_r2r(const fftwq_plan p, __float128 *in, __float128 *out); 
extern void fftwq_destroy_plan(fftwq_plan p); 
extern void fftwq_forget_wisdom(void); 
extern void fftwq_cleanup(void); 
extern void fftwq_set_timelimit(double t); 
extern void fftwq_plan_with_nthreads(int nthreads); 
extern int fftwq_init_threads(void); 
extern void fftwq_cleanup_threads(void); 
extern int fftwq_export_wisdom_to_filename(const char *filename); 
extern void fftwq_export_wisdom_to_file(FILE *output_file); 
extern char *fftwq_export_wisdom_to_string(void); 
extern void fftwq_export_wisdom(fftwq_write_char_func write_char, void *data); 
extern int fftwq_import_system_wisdom(void); 
extern int fftwq_import_wisdom_from_filename(const char *filename); 
extern int fftwq_import_wisdom_from_file(FILE *input_file); 
extern int fftwq_import_wisdom_from_string(const char *input_string); 
extern int fftwq_import_wisdom(fftwq_read_char_func read_char, void *data); 
extern void fftwq_fprint_plan(const fftwq_plan p, FILE *output_file); 
extern void fftwq_print_plan(const fftwq_plan p); 
extern void *fftwq_malloc(size_t n); 
extern __float128 *fftwq_alloc_real(size_t n); 
extern fftwq_complex *fftwq_alloc_complex(size_t n); 
extern void fftwq_free(void *p); 
extern void fftwq_flops(const fftwq_plan p, double *add, double *mul, double *fmas); 
extern double fftwq_estimate_cost(const fftwq_plan p); 
extern double fftwq_cost(const fftwq_plan p); 
extern const char fftwq_version[]; 
extern const char fftwq_cc[]; 
extern const char fftwq_codelet_optim[];

]]
