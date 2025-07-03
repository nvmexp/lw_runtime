#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <stdint.h>
#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBLwrsor : public Cursor {
 public:
  explicit LMDBLwrsor(MDB_txn* mdb_txn, MDB_lwrsor* mdb_lwrsor)
    : mdb_txn_(mdb_txn), mdb_lwrsor_(mdb_lwrsor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBLwrsor() {
    mdb_lwrsor_close(mdb_lwrsor_);
    mdb_txn_abort(mdb_txn_);
  }
  void SeekToFirst() override { Seek(MDB_FIRST); }
  void Next() override { Seek(MDB_NEXT); }
  string key() const override {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  string value() const override {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  bool parse(Datum* datum) const override {
    return datum->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  }
  bool parse(AnnotatedDatum* adatum) const override {
    return adatum->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  }
  bool parse(C2TensorProtos* c2p) const override {
    return c2p->ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  }
  const void* data() const override {
    return mdb_value_.mv_data;
  }
  size_t size() const override {
    return mdb_value_.mv_size;
  }

  bool valid() const override { return valid_; }

 private:
  void Seek(MDB_lwrsor_op op) {
    int mdb_status = mdb_lwrsor_get(mdb_lwrsor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  MDB_txn* mdb_txn_;
  MDB_lwrsor* mdb_lwrsor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_elw* mdb_elw)
    : mdb_elw_(mdb_elw) { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  MDB_elw* mdb_elw_;
  vector<string> keys, values;

  void DoubleMapSize();

  DISABLE_COPY_MOVE_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_elw_(NULL), mdb_dbi_() { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_elw_ != NULL) {
      mdb_dbi_close(mdb_elw_, mdb_dbi_);
      mdb_elw_close(mdb_elw_);
      mdb_elw_ = NULL;
    }
  }
  virtual LMDBLwrsor* NewLwrsor();
  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_elw* mdb_elw_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
